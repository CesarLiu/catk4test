import numpy as np

from src.responsibility.hmm import GaussianHMM, fit_hmm_with_model_selection


def _make_two_level_sequences(n_seq=60, seq_len=20, seed=0):
    rng = np.random.default_rng(seed)
    means = np.array([[0.0, 0.0], [8.0, 8.0]])
    sequences, true_levels = [], []
    for _ in range(n_seq):
        levels = rng.integers(0, 2, size=seq_len)
        obs = means[levels] + rng.normal(scale=0.3, size=(seq_len, 2))
        sequences.append(obs)
        true_levels.append(levels)
    return sequences, true_levels


def test_gaussian_hmm_fit_recovers_well_separated_means():
    sequences, _ = _make_two_level_sequences()
    hmm = GaussianHMM(n_states=2, n_features=2, seed=0)
    history = hmm.fit(sequences, n_iter=50)
    assert len(history) > 0
    # loglik should improve (non-decreasing) apart from the first couple of steps
    assert history[-1] >= history[0]

    means_sorted = np.sort(hmm.means_.sum(axis=1))
    assert means_sorted[0] < 4.0  # near the (0,0) cluster
    assert means_sorted[1] > 12.0  # near the (8,8) cluster


def test_gaussian_hmm_forward_filter_matches_generating_level():
    rng = np.random.default_rng(1)
    means = np.array([[0.0, 0.0], [8.0, 8.0]])
    sequences, true_levels = _make_two_level_sequences(n_seq=40, seed=2)
    hmm = GaussianHMM(n_states=2, n_features=2, seed=3)
    hmm.fit(sequences, n_iter=50)

    # relabel predicted states to match the generating labels via mean matching
    pred_to_true = {}
    for h in range(2):
        pred_to_true[h] = int(np.argmin(np.linalg.norm(means - hmm.means_[h], axis=1)))

    correct, total = 0, 0
    for obs, levels in zip(sequences, true_levels):
        pred = hmm.predict_level(obs)
        pred_mapped = np.array([pred_to_true[p] for p in pred])
        correct += (pred_mapped == levels).sum()
        total += len(levels)
    assert correct / total > 0.85


def test_bic_prefers_correct_number_of_states():
    sequences, _ = _make_two_level_sequences(n_seq=80, seed=4)
    result = fit_hmm_with_model_selection(
        sequences, n_states_grid=(1, 2, 3, 5), n_iter=50, seed=5
    )
    assert result.best_n_states == 2


def test_bic_and_score_are_finite():
    sequences, _ = _make_two_level_sequences(n_seq=10, seed=6)
    hmm = GaussianHMM(n_states=2, n_features=2, seed=7)
    hmm.fit(sequences, n_iter=20)
    bic = hmm.bic(sequences)
    assert np.isfinite(bic)
    for obs in sequences:
        assert np.isfinite(hmm.score(obs))


def test_forward_filter_is_a_valid_distribution():
    sequences, _ = _make_two_level_sequences(n_seq=5, seed=8)
    hmm = GaussianHMM(n_states=3, n_features=2, seed=9)
    hmm.fit(sequences, n_iter=20)
    posterior = hmm.forward_filter(sequences[0])
    assert posterior.shape == (sequences[0].shape[0], 3)
    np.testing.assert_allclose(posterior.sum(axis=-1), 1.0, atol=1e-6)
    assert (posterior >= 0).all()
