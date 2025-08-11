from __future__ import annotations
import json
from pathlib import Path
from typing import Tuple
import numpy as np
from scipy.signal import welch


def save_psd_csv(t: np.ndarray, x: np.ndarray, out_csv: str, fs: float | None = None,
                  nperseg: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    """Compute and save Welch PSD (frequency, power) to CSV."""
    if fs is None:
        dt = float(np.median(np.diff(t)))
        fs = 1.0 / dt
    f, Pxx = welch(x, fs=fs, nperseg=min(nperseg, len(x)))
    arr = np.column_stack([f, Pxx])
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_csv, arr, delimiter=",", header="freq_Hz,PSD", comments="")
    return f, Pxx


def save_order_parameter_csv(t: np.ndarray, r: np.ndarray, out_csv: str) -> None:
    arr = np.column_stack([t, r])
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_csv, arr, delimiter=",", header="t_s,r", comments="")


def save_metadata_json(meta: dict, out_json: str) -> None:
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(meta, indent=2))
