from __future__ import annotations
import numpy as np
from .config import DriveConfig

class ProgrammableDrive:
    """
    J(t) = sum_s A_s * env(t) * cos(2π f_s t + φ_s)
    Spatial mask kept symbolic via metadata (U(x)) for now.
    """
    def __init__(self, cfg: DriveConfig):
        self.cfg = cfg
        self._tones = np.array([(t.freq, t.amp, t.phase) for t in cfg.tones], dtype=float)

    def envelope(self, t: np.ndarray) -> np.ndarray:
        if self.cfg.envelope.kind == "hann":
            T = self.cfg.duration
            env = 0.5 * (1 - np.cos(2 * np.pi * np.clip(t, 0, T) / T))
            env[(t < 0) | (t > T)] = 0.0
            return env
        if self.cfg.envelope.kind == "rect":
            return ((t >= 0) & (t <= self.cfg.duration)).astype(float)
        if self.cfg.envelope.kind == "exp":
            tau = max(self.cfg.envelope.tau, 1e-6)
            env = 1 - np.exp(-np.clip(t, 0, None) / tau)
            env[(t < 0) | (t > self.cfg.duration)] = 0.0
            return env
        raise ValueError("Unknown envelope kind")

    def time_series(self) -> tuple[np.ndarray, np.ndarray]:
        dt, T = self.cfg.dt, self.cfg.duration
        t = np.arange(0.0, T, dt)
        env = self.envelope(t)
        J = np.zeros_like(t)
        for f, A, phi in self._tones:
            J += A * env * np.cos(2 * np.pi * f * t + phi)
        return t, J
