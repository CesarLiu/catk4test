"""Gaussian hidden Markov model for responsibility-level abstraction (paper
Sec. IV-A). A small from-scratch Baum-Welch implementation (rather than an
extra third-party HMM dependency): responsibility sequences are low
dimensional (D=2: safety, courtesy) so a NumPy EM loop is simple and fast
enough, and it keeps this repo's dependency footprint unchanged (numpy and
scipy are already transitive dependencies here).
"""

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np
from scipy.special import logsumexp


class GaussianHMM:
    """HMM with H hidden responsibility levels and a diagonal-covariance
    Gaussian emission per level, ``b_h(beta) = N(beta | mu_h, Sigma_h)``,
    matching the paper's (rho, A, B) parameterization (Sec. IV-A)."""

    def __init__(self, n_states: int, n_features: int, min_covar: float = 1e-3, seed: int = 0):
        self.n_states = n_states
        self.n_features = n_features
        self.min_covar = min_covar
        self.rng = np.random.default_rng(seed)
        self.startprob_ = np.full(n_states, 1.0 / n_states)
        self.transmat_ = np.full((n_states, n_states), 1.0 / n_states)
        self.means_ = np.zeros((n_states, n_features))
        self.covars_ = np.ones((n_states, n_features))

    def _init_params(self, sequences: Sequence[np.ndarray]) -> None:
        all_obs = np.concatenate(sequences, axis=0)
        idx = self.rng.choice(len(all_obs), size=self.n_states, replace=len(all_obs) < self.n_states)
        self.means_ = all_obs[idx].astype(np.float64).copy()
        var = all_obs.var(axis=0) + self.min_covar
        self.covars_ = np.tile(var, (self.n_states, 1))
        self.startprob_ = np.full(self.n_states, 1.0 / self.n_states)
        self.transmat_ = np.full((self.n_states, self.n_states), 1.0 / self.n_states)

    def _log_emission(self, obs: np.ndarray) -> np.ndarray:  # [T, D] -> [T, H]
        diff = obs[:, None, :] - self.means_[None, :, :]  # [T, H, D]
        var = np.clip(self.covars_, self.min_covar, None)  # [H, D]
        log_norm = -0.5 * np.sum(np.log(2 * np.pi * var), axis=-1)  # [H]
        quad = -0.5 * np.sum(diff**2 / var[None, :, :], axis=-1)  # [T, H]
        return quad + log_norm[None, :]

    def _forward_backward(self, log_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        T, H = log_b.shape
        log_start = np.log(self.startprob_ + 1e-300)
        log_trans = np.log(self.transmat_ + 1e-300)

        log_alpha = np.zeros((T, H))
        log_alpha[0] = log_start + log_b[0]
        for t in range(1, T):
            log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_trans, axis=0) + log_b[t]

        log_beta = np.zeros((T, H))
        for t in range(T - 2, -1, -1):
            log_beta[t] = logsumexp(log_trans + (log_b[t + 1] + log_beta[t + 1])[None, :], axis=1)

        loglik = float(logsumexp(log_alpha[-1]))
        log_gamma = log_alpha + log_beta - loglik
        gamma = np.exp(log_gamma)

        if T > 1:
            log_xi = (
                log_alpha[:-1, :, None]
                + log_trans[None, :, :]
                + log_b[1:, None, :]
                + log_beta[1:, None, :]
                - loglik
            )
            xi = np.exp(log_xi)  # [T-1, H, H]
        else:
            xi = np.zeros((0, H, H))
        return gamma, xi, loglik

    def fit(self, sequences: Sequence[np.ndarray], n_iter: int = 100, tol: float = 1e-4, verbose: bool = False) -> List[float]:
        sequences = [np.asarray(s, dtype=np.float64) for s in sequences if len(s) > 0]
        self._init_params(sequences)

        history = []
        prev_ll = -np.inf
        for it in range(n_iter):
            start_acc = np.zeros(self.n_states)
            trans_num = np.zeros((self.n_states, self.n_states))
            trans_den = np.zeros(self.n_states)
            mean_num = np.zeros((self.n_states, self.n_features))
            mean_den = np.zeros(self.n_states)
            total_ll = 0.0
            gammas = []

            for obs in sequences:
                log_b = self._log_emission(obs)
                gamma, xi, ll = self._forward_backward(log_b)
                gammas.append(gamma)
                total_ll += ll
                start_acc += gamma[0]
                if len(obs) > 1:
                    trans_num += xi.sum(axis=0)
                    trans_den += gamma[:-1].sum(axis=0)
                mean_num += gamma.T @ obs
                mean_den += gamma.sum(axis=0)

            history.append(total_ll)
            if verbose:
                print(f"[GaussianHMM] iter {it}: loglik={total_ll:.4f}")
            if abs(total_ll - prev_ll) < tol * max(1.0, abs(prev_ll)):
                break
            prev_ll = total_ll

            self.startprob_ = start_acc / max(start_acc.sum(), 1e-12)
            new_transmat = trans_num / np.clip(trans_den[:, None], 1e-12, None)
            flat_rows = trans_den < 1e-12
            new_transmat[flat_rows] = 1.0 / self.n_states
            self.transmat_ = new_transmat

            new_means = mean_num / np.clip(mean_den[:, None], 1e-12, None)
            empty = mean_den < 1e-12
            new_means[empty] = self.means_[empty]

            cov_num = np.zeros((self.n_states, self.n_features))
            for obs, gamma in zip(sequences, gammas):
                diff = obs[:, None, :] - new_means[None, :, :]  # [T, H, D]
                cov_num += np.einsum("th,thd->hd", gamma, diff**2)
            new_covars = cov_num / np.clip(mean_den[:, None], 1e-12, None)
            new_covars = np.clip(new_covars, self.min_covar, None)
            new_covars[empty] = self.covars_[empty]

            self.means_ = new_means
            self.covars_ = new_covars

        return history

    def score(self, obs: np.ndarray) -> float:
        """Total log-likelihood of one sequence, log p(beta_1:T)."""
        obs = np.asarray(obs, dtype=np.float64)
        if len(obs) == 0:
            return 0.0
        log_b = self._log_emission(obs)
        _, _, ll = self._forward_backward(log_b)
        return ll

    def n_free_params(self) -> int:
        return (
            (self.n_states - 1)
            + self.n_states * (self.n_states - 1)
            + self.n_states * self.n_features  # means
            + self.n_states * self.n_features  # diagonal covariances
        )

    def bic(self, sequences: Sequence[np.ndarray]) -> float:
        """BIC := 2*NLL + m*log(n), matching the paper's Sec. V-B definition."""
        total_ll = sum(self.score(obs) for obs in sequences)
        n_obs = sum(len(obs) for obs in sequences)
        if n_obs == 0:
            return float("inf")
        return 2.0 * (-total_ll) + self.n_free_params() * np.log(n_obs)

    def forward_filter(self, obs: np.ndarray) -> np.ndarray:  # [T, H]
        """Online Bayes-filter posterior p(z_k | beta_{1:k}), Eq. (6)."""
        obs = np.asarray(obs, dtype=np.float64)
        log_b = self._log_emission(obs)
        T, H = log_b.shape
        log_start = np.log(self.startprob_ + 1e-300)
        log_trans = np.log(self.transmat_ + 1e-300)

        log_filt = np.zeros((T, H))
        log_joint = log_start + log_b[0]
        log_filt[0] = log_joint - logsumexp(log_joint)
        for t in range(1, T):
            log_predict = logsumexp(log_filt[t - 1][:, None] + log_trans, axis=0)
            log_joint = log_predict + log_b[t]
            log_filt[t] = log_joint - logsumexp(log_joint)
        return np.exp(log_filt)

    def predict_level(self, obs: np.ndarray) -> np.ndarray:  # [T] int
        """z_k = argmax_z p(z | beta_{1:k}) at every step, Eq. (6)."""
        return self.forward_filter(obs).argmax(axis=-1)


@dataclass
class ModelSelectionResult:
    n_states_grid: List[int]
    bic_by_n_states: List[float]
    best_n_states: int
    best_model: GaussianHMM = field(repr=False)


def fit_hmm_with_model_selection(
    sequences: Sequence[np.ndarray],
    n_states_grid: Sequence[int] = (3, 5, 7, 9, 11),
    n_features: int = 2,
    n_iter: int = 100,
    tol: float = 1e-4,
    seed: int = 0,
    verbose: bool = False,
) -> ModelSelectionResult:
    """Sweeps the number of responsibility levels H and picks the model
    with the lowest BIC, reproducing the paper's Fig. 3 (left) selection."""
    bics = []
    models = []
    for h in n_states_grid:
        model = GaussianHMM(n_states=h, n_features=n_features, seed=seed)
        model.fit(sequences, n_iter=n_iter, tol=tol, verbose=verbose)
        bic = model.bic(sequences)
        bics.append(bic)
        models.append(model)
        if verbose:
            print(f"[fit_hmm_with_model_selection] H={h}: BIC={bic:.2f}")

    best_i = int(np.argmin(bics))
    return ModelSelectionResult(
        n_states_grid=list(n_states_grid),
        bic_by_n_states=bics,
        best_n_states=n_states_grid[best_i],
        best_model=models[best_i],
    )
