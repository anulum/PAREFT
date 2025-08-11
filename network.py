from __future__ import annotations
import numpy as np
from numpy.random import default_rng
from dataclasses import dataclass

@dataclass
class NetworkState:
    theta: np.ndarray  # phases
    omega: np.ndarray  # natural frequencies

class KuramotoInertial:
    """Kuramoto model with inertia, damping, delays (single global delay), and noise.

    M θ̈_i + γ θ̇_i = ω_i + (K/N) Σ_j sin(θ_j(t-τ) - θ_i(t)) + F sin(Ω t - θ_i + φ) + ξ_i(t)
    """
    def __init__(self, N: int, K: float, M: float, gamma: float, omega_spread: float,
                 delay: float = 0.0, noise_std: float = 0.0, seed: int = 1234):
        self.N, self.K, self.M, self.gamma = N, K, M, gamma
        self.delay, self.noise_std = delay, noise_std
        self.rng = default_rng(seed)
        self.state = NetworkState(
            theta=2*np.pi*self.rng.random(N),
            omega=self.rng.normal(0.0, omega_spread, size=N),
        )

    def order_parameter(self, theta: np.ndarray) -> complex:
        return np.mean(np.exp(1j*theta))

    def simulate(self, t: np.ndarray, drive: dict) -> dict:
        dt = float(np.median(np.diff(t)))
        F, Omega, phi = drive["F"], drive["Omega"], drive.get("phi", 0.0)
        theta = self.state.theta.copy()
        vel = np.zeros_like(theta)
        r_hist = np.zeros_like(t, dtype=float)

        # Simple ring buffer for delay (uniform τ)
        max_delay_steps = int(max(self.delay, 0.0) / dt)
        theta_hist = np.tile(theta, (max(1, max_delay_steps)+1, 1))
        idx = 0

        for k, tk in enumerate(t):
            # Delayed mean field (all-to-all simplifying assumption)
            if max_delay_steps > 0:
                theta_delayed = theta_hist[(idx - max_delay_steps) % theta_hist.shape[0]]
                r = np.mean(np.exp(1j*theta_delayed))
            else:
                r = np.mean(np.exp(1j*theta))
            r_hist[k] = np.abs(r)

            coupling = self.K * np.abs(r) * np.sin(np.angle(r) - theta)  # MF reduction
            drive_term = F * np.sin(Omega * tk - theta + phi)
            noise = self.noise_std * self.rng.standard_normal(self.N)

            acc = (self.state.omega + coupling + drive_term + noise - self.gamma*vel) / self.M
            vel += dt * acc
            theta = (theta + dt * vel) % (2*np.pi)

            # update ring buffer
            if max_delay_steps > 0:
                idx = (idx + 1) % theta_hist.shape[0]
                theta_hist[idx] = theta

        return {"r": r_hist, "theta": theta}
