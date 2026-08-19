"""
Extraction des 12 features temporelles a partir des probabilites frame-level
d'une video. Copie EXACTE de `extract_temporal_features` du notebook
d'entrainement (Cellule 12), pour que le classifieur temporel (entraine sur
ces memes features) reste valide en inference.
"""
import numpy as np
import scipy.stats
from scipy.signal import find_peaks

FEATURE_NAMES = [
    "mean_prob", "std_prob", "max_prob", "min_prob", "range_prob",
    "skewness", "kurtosis", "diff_mean", "diff_std",
    "n_peaks_norm", "n_drops_norm", "trend",
]


def extract_temporal_features(frame_probs):
    if len(frame_probs) < 2:
        v = frame_probs[0]
        return np.array([v, 0, v, v, 0, 0, 0, 0, 0, 0, 0, 0])

    probs = np.array(frame_probs)
    n_frames = len(probs)
    mean_prob, std_prob = np.mean(probs), np.std(probs)
    max_prob, min_prob = np.max(probs), np.min(probs)
    range_prob = max_prob - min_prob
    skewness = scipy.stats.skew(probs)
    kurtosis = scipy.stats.kurtosis(probs)
    diffs = np.diff(probs)
    diff_mean, diff_std = np.mean(diffs), np.std(diffs)
    peaks, _ = find_peaks(probs, height=0.7)
    drops, _ = find_peaks(1 - probs, height=0.7)
    n_peaks_norm, n_drops_norm = len(peaks) / n_frames, len(drops) / n_frames
    trend = np.polyfit(range(len(probs)), probs, 1)[0] if len(probs) > 1 else 0
    features = np.array([mean_prob, std_prob, max_prob, min_prob, range_prob,
                          skewness, kurtosis, diff_mean, diff_std,
                          n_peaks_norm, n_drops_norm, trend])
    return np.nan_to_num(features, nan=0.0)
