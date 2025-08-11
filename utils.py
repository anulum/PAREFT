from __future__ import annotations
import numpy as np

def plv(x_phase: np.ndarray, y_phase: np.ndarray) -> float:
    """Phase-locking value between two phase time series."""
    return np.abs(np.mean(np.exp(1j*(x_phase - y_phase))))


def analytic_phase(x: np.ndarray) -> np.ndarray:
    """Hilbert analytic signal phase."""
    from scipy.signal import hilbert
    return np.angle(hilbert(x))
