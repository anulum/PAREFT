from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from .config import EFTConfig

class Susceptibility:
    """Loop-aware placeholder for χ̃_full(ω).

    Model a damped oscillator Green's function dressed by a simple
    frequency-dependent self-energy Σ(ω) ~ λ_phi^2 * (ω^2)/(ω^2 + m_phi^2).
    """
    def __init__(self, cfg: EFTConfig):
        self.cfg = cfg

    def chi0(self, omega: ArrayLike) -> np.ndarray:
        m, gamma = self.cfg.m_phi, self.cfg.damping
        return 1.0 / (m**2 - (omega**2) - 1j * gamma * omega)

    def self_energy(self, omega: ArrayLike) -> np.ndarray:
        lam, m = self.cfg.lambda_phi, self.cfg.m_phi
        return (lam**2) * (omega**2) / (omega**2 + m**2 + 1e-12)

    def chi_full(self, omega: ArrayLike) -> np.ndarray:
        chi0 = self.chi0(omega)
        Sigma = self.self_energy(omega)
        return chi0 / (1.0 - Sigma * chi0)

    def response(self, omega: ArrayLike, Jw: ArrayLike) -> np.ndarray:
        """Frequency-domain response Φ(ω) ≈ χ̃_full(ω) J(ω)."""
        return self.chi_full(omega) * Jw
