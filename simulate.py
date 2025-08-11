from __future__ import annotations
from pathlib import Path
import numpy as np
from .config import ExperimentConfig
from .drive import ProgrammableDrive
from .eft import Susceptibility
from .network import KuramotoInertial
from .probes import save_psd_csv, save_order_parameter_csv, save_metadata_json


def run_experiment(cfg: ExperimentConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)

    # Drive
    drv = ProgrammableDrive(cfg.drive)
    t, Jt = drv.time_series()

    # Frequency-domain response (symbolic single-mode placeholder)
    chi = Susceptibility(cfg.eft)
    # FFT (rfft for real signals)
    dt = cfg.drive.dt
    Jw = np.fft.rfft(Jt)
    freqs = np.fft.rfftfreq(len(Jt), d=dt) * 2*np.pi  # rad/s
    Phi_w = chi.response(freqs, Jw)
    phi_t = np.fft.irfft(Phi_w, n=len(Jt))  # induced field proxy (scalar channel)

    # Network dynamics (phase-locked ensemble) driven at dominant tone
    # Choose the highest amplitude tone as Ω
    tones = sorted(cfg.drive.tones, key=lambda x: x.amp, reverse=True)
    Omega = 2*np.pi*tones[0].freq if tones else 0.0
    F = tones[0].amp if tones else 0.0
    net = KuramotoInertial(
        N=cfg.network.N, K=cfg.network.K, M=cfg.network.M, gamma=cfg.network.gamma,
        omega_spread=cfg.network.omega_spread, delay=cfg.network.delay,
        noise_std=cfg.network.noise_std, seed=cfg.seed,
    )
    res = net.simulate(t, {"F": F, "Omega": Omega, "phi": tones[0].phase if tones else 0.0})

    # Outputs
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Save PSDs for J(t) and φ(t)
    save_psd_csv(t, Jt, str(outdir/"psd_J.csv"))
    save_psd_csv(t, phi_t, str(outdir/"psd_phi.csv"))

    # Save order parameter r(t)
    save_order_parameter_csv(t, res["r"], str(outdir/"order_parameter.csv"))

    # Metadata
    meta = {
        "cfg": cfg.__dict__,
        "tones": [t.__dict__ for t in cfg.drive.tones],
        "notes": "PAREFT scaffold v0.1",
    }
    save_metadata_json(meta, str(outdir/"metadata.json"))

    return {"t": t, "Jt": Jt, "phi_t": phi_t, "r": res["r"]}
