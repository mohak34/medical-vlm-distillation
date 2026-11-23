"""AUROC with tie handling, patient-level bootstrap, and summary helpers."""

import numpy as np
from sklearn.metrics import roc_auc_score

N_BOOT = 1000
BOOT_SEED = 0


def auroc(y: np.ndarray, s: np.ndarray, w: np.ndarray | None = None) -> float:
    """Weighted AUROC (Mann-Whitney with half credit for ties). Weight w duplicates a row w times."""
    w = np.ones(len(y)) if w is None else w
    _, inv = np.unique(s, return_inverse=True)
    pos = np.bincount(inv, weights=w * (y == 1))
    neg = np.bincount(inv, weights=w * (y == 0))
    below = np.cumsum(neg) - neg  # negatives with strictly lower score
    return float((pos * (below + 0.5 * neg)).sum() / (pos.sum() * neg.sum()))


def auroc_table(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Per-column AUROC for (N,F) labels and scores."""
    return np.array([roc_auc_score(y[:, f], s[:, f]) for f in range(y.shape[1])])


def boot_weights(patients: np.ndarray, n: int = N_BOOT, seed: int = BOOT_SEED) -> np.ndarray:
    """(n, N) row weights: resample patients with replacement, every row of a patient gets its count.
    Same seed -> same resamples, so differences between models are paired."""
    uniq, idx = np.unique(patients, return_inverse=True)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=n)
    return counts[:, idx].astype(np.float64)


def boot_auroc(y: np.ndarray, s: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """(n_boot, F) AUROCs under each resample."""
    return np.array([[auroc(y[:, f], s[:, f], w) for f in range(y.shape[1])] for w in weights])


def ci(samples: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    return np.percentile(samples, 2.5, axis=axis), np.percentile(samples, 97.5, axis=axis)
