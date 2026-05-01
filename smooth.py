import numpy as np


def smooth_landmarks(curr, prev, alpha=0.4):

    # First frame OR face count changed
    if prev is None or prev.shape != curr.shape:
        return curr, curr.copy()

    # Exponential smoothing (vectorized)
    smoothed = alpha * curr + (1 - alpha) * prev

    return smoothed, smoothed.copy()