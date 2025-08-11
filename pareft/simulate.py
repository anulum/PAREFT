from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
import numpy as np
import matplotlib.pyplot as plt
from .config import ExperimentConfig
from .drive import ProgrammableDrive
from .eft import Susceptibility
from .network import KuramotoInertial
from .probes import save_psd_csv, save_order_parameter_csv, save_metadata_json


def run_experiment(cfg: ExperimentConfig, *, save_raw: bool = True, save_plots: bool = True) -> dict:
    rng = np.random.default_rng(cfg.seed)

    # Drive
    drv = ProgrammableDrive(cfg.drive)
    t, Jt = drv.time_series()

    # Frequency-domain response (symbolic single-mode placeholder)
    chi = Susceptibility(cfg.eft)
    dt = cfg.drive.dt
    Jw = np.fft.rfft(Jt)
    freqs = np.fft.rfftfreq(len(Jt), d=dt) * 2*np.pi  # rad/s
    Phi_w = chi.response(freqs, Jw)
    phi_t = np.fft.irfft(Phi_w, n=len(Jt))  # induced field proxy (scalar channel)

    # Network dynamics (phase-locked ensemble) driven at dominant tone
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
    fJ, PJ = save_psd_csv(t, Jt, str(outdir/"psd_J.csv"))
    fP, PP = save_psd_csv(t, phi_t, str(outdir/"psd_phi.csv"))

    # Save order parameter r(t)
    save_order_parameter_csv(t, res["r"], str(outdir/"order_parameter.csv"))

    # Save raw timeseries (t,J,phi,r) – podľa flagu
    if save_raw:
        ts = np.column_stack([t, Jt, phi_t, res["r"]])
        np.savetxt(outdir/"timeseries.csv", ts, delimiter=",", header="t,J,phi,r", comments="")

    # Auto-plots počas simulácie – podľa flagu
    if save_plots:
        # PSD J
        fig1, ax1 = plt.subplots()
        ax1.loglog(fJ, PJ)
        ax1.set_xlabel("f [Hz]")
        ax1.set_ylabel("PSD[J]")
        ax1.set_title("Drive PSD")
        fig1.tight_layout()
        fig1.savefig(outdir/"plot_1.png", dpi=150)
        plt.close(fig1)

        # PSD phi
        fig2, ax2 = plt.subplots()
        ax2.loglog(fP, PP)
        ax2.set_xlabel("f [Hz]")
        ax2.set_ylabel("PSD[phi]")
        ax2.set_title("Induced Field PSD")
        fig2.tight_layout()
        fig2.savefig(outdir/"plot_2.png", dpi=150)
        plt.close(fig2)

        # r(t)
        fig3, ax3 = plt.subplots()
        ax3.plot(t, res["r"])
        ax3.set_xlabel("t [s]")
        ax3.set_ylabel("r(t)")
        ax3.set_title("Order parameter")
        fig3.tight_layout()
        fig3.savefig(outdir/"plot_3.png", dpi=150)
        plt.close(fig3)

    # Metadata (JSON-serializovateľné)
    meta = {
        "cfg": asdict(cfg),
        "tones": [asdict(tone) for tone in cfg.drive.tones],
        "notes": "PAREFT scaffold v0.1",
        "save_raw": bool(save_raw),
        "save_plots": bool(save_plots),
    }
    save_metadata_json(meta, str(outdir/"metadata.json"))

    return {"t": t, "Jt": Jt, "phi_t": phi_t, "r": res["r"]}
