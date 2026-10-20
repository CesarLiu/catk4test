"""Flags a scene by its single worst stretch of responsibility.

One agent's beta sequence (one value per context window, ``window_stride``
steps apart) is first smoothed over time, then a sliding window of a given
length is moved along it; the scene's score per channel is the worst window
(largest mean beta_s: the most dangerous stretch; largest mean beta_c: the
least courteous). One bad stretch is enough to flag the scene, however calm
the rest of the clip is.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SMOOTHERS = ("none", "moving_average", "gaussian", "ema")
CHANNELS = ("safety", "courtesy", "either")


@dataclass
class FlagConfig:
    smoother: str = "gaussian"
    # seconds: the Gaussian's sigma, the moving average's full width, or the
    # EMA's time constant
    smooth_width: float = 0.5
    # seconds of consecutive windows averaged into one score; the window
    # spacing (0.5 s by default) is the shortest
    window: float = 1.0
    channel: str = "either"
    safety_threshold: float = 0.5  # m
    courtesy_threshold: float = 1.0  # nats


def smooth(x: np.ndarray, dt: float, smoother: str, width: float) -> np.ndarray:
    """Smooths a [T] series sampled every ``dt`` seconds. Near the ends the
    kernel is renormalised over the samples that exist, so the first and
    last windows are not pulled toward zero."""
    if smoother not in SMOOTHERS:
        raise ValueError(f"smoother must be one of {SMOOTHERS}, got {smoother!r}")
    x = np.asarray(x, dtype=float)
    if smoother == "none" or width <= 0 or len(x) < 2:
        return x.copy()
    if smoother == "ema":  # causal: usable online
        a = 1.0 - np.exp(-dt / width)
        out = np.empty_like(x)
        out[0] = x[0]
        for t in range(1, len(x)):
            out[t] = out[t - 1] + a * (x[t] - out[t - 1])
        return out
    lags = np.arange(len(x))[:, None] - np.arange(len(x))[None, :]  # [T, T]
    if smoother == "gaussian":
        w = np.exp(-0.5 * (lags * dt / width) ** 2)
    else:
        w = (np.abs(lags * dt) <= width / 2 + 1e-9).astype(float)
    return (w @ x) / w.sum(axis=1)


def worst_window(x: np.ndarray, n: int) -> Tuple[float, int]:
    """(largest mean over ``n`` consecutive samples, its first index); a
    sequence shorter than ``n`` is one window."""
    n = max(1, min(n, len(x)))
    means = np.convolve(x, np.ones(n) / n, mode="valid")
    i = int(np.argmax(means))
    return float(means[i]), i


def window_count(window: float, dt: float) -> int:
    return max(1, int(round(window / dt)))


def scene_score(ks: Sequence[int], beta: np.ndarray, cfg: FlagConfig, step_dt: float = 0.1) -> Dict:
    """Worst-window scores of one agent's sequence. ``ks`` are the windows'
    current steps (10 Hz), ``beta`` [T, 2] their (safety, courtesy)."""
    if cfg.channel not in CHANNELS:
        raise ValueError(f"channel must be one of {CHANNELS}, got {cfg.channel!r}")
    ks = np.asarray(ks)
    beta = np.asarray(beta, dtype=float).reshape(len(ks), 2)
    dt = float(np.median(np.diff(ks))) * step_dt if len(ks) > 1 else 0.5
    n = window_count(cfg.window, dt)
    out = {"n_windows": len(ks)}
    for d, name in enumerate(("safety", "courtesy")):
        s = smooth(beta[:, d], dt, cfg.smoother, cfg.smooth_width)
        value, i = worst_window(s, n)
        j = min(i + n, len(ks)) - 1
        out[name] = value
        # the stretch, in seconds of context time t_k
        out[f"{name}_t"] = (float(ks[i]) * step_dt, float(ks[j]) * step_dt)
    # either channel, each in units of its own threshold
    out["either"] = max(out["safety"] / cfg.safety_threshold, out["courtesy"] / cfg.courtesy_threshold)
    out["score"] = out[cfg.channel]
    threshold = 1.0 if cfg.channel == "either" else getattr(cfg, f"{cfg.channel}_threshold")
    out["flagged"] = bool(out["score"] >= threshold)
    return out


def score_sequences(grouped: Dict[Tuple, List], cfg: FlagConfig) -> Dict[Tuple, Dict]:
    """``grouped`` is dataset_builder.group_sequences' output."""
    return {
        key: scene_score([o.k for o in obs], np.stack([o.beta.numpy() for o in obs]), cfg)
        for key, obs in grouped.items()
        if obs
    }


def ranking_metrics(scores: Sequence[float], labels: Sequence[int], flagged: Optional[Sequence[bool]] = None) -> Dict:
    """AUROC, average precision and precision at the number of positives;
    with ``flagged``, precision and recall of the flags."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    s, y = np.asarray(scores, dtype=float), np.asarray(labels, dtype=int)
    p = int(y.sum())
    out = {"n": len(y), "positives": p}
    if 0 < p < len(y):
        out["auroc"] = float(roc_auc_score(y, s))
        out["ap"] = float(average_precision_score(y, s))
        out[f"precision@{p}"] = float(y[np.argsort(-s)[:p]].mean())
    if flagged is not None:
        f = np.asarray(flagged, dtype=bool)
        out["flagged"] = int(f.sum())
        out["precision"] = float(y[f].mean()) if f.any() else float("nan")
        out["recall"] = float(f[y == 1].mean()) if p else float("nan")
    return out
