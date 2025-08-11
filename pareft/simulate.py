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


def _save_plots(outdir: Path, psd_J, psd_phi, t, r_t) -> None:
    # Drive PSD
    fig1, ax1 = plt.subplots()
    ax1.loglog(psd_J[0], psd_J[1])
    ax1.set_xlabel("f [Hz]")
    ax1.set_ylabel("PSD[J]")
    ax1.set_title("Drive PSD")
    fig1.tight_layout()
    fig1.savefig(outdir / "plot_drive_psd.png", dpi=150)
    plt.close(fig1)

    # Phi PSD
    fig2, ax2 = plt.subplots()
    ax2.loglog(psd_phi[0], psd_phi[1])
    ax2.set_xlabel("f [Hz]")
    ax2.set_ylabel("PSD[phi]")
    ax2.set_title("Induced Field PSD (proxy)")
    fig2.tight_layout()
    fig2.savefig(outdir / "plot_phi_psd.png", dpi=150)
    plt.close(fig2)

    # Order parameter r(t)
    fig3, ax3 = plt.subplots()
    ax3.plot(t, r_t)
    ax3.set_xlabel("t [s]")
    ax3.set_ylabel("r(t)")
    ax3.set_title("Order parameter")
    fig3.tight_layout()
    fig3.savefig(outdir / "plot_order_parameter.png", dpi=150)
    plt.close(fig3)


def run_experiment(cfg: ExperimentConfig, *, save_raw: bool = False, save_png: bool = True) -> dict:
    rng = np.random.default_rng(cfg.seed)

    # Drive
    drv = ProgrammableDrive(cfg.drive)
    t, Jt = drv.time_series()

    # Frequency-domain response (symbolic single-mode placeholder)
    chi = Susceptibility(cfg.eft)
    dt = cfg.drive.dt
    Jw = np.fft.rfft(Jt)
    freqs = np.fft.rfftfreq(len(Jt), d=dt) * 2 * np.pi  # rad/s
    Phi_w = chi.response(freqs, Jw)
    phi_t = np.fft.irfft(Phi_w, n=len(Jt))  # induced field proxy

    # Network dynamics (phase-locked ensemble) driven na dominantnom tóne
    tones = sorted(cfg.drive.tones, key=lambda x: x.amp, reverse=True)
    Omega = 2 * np.pi * tones[0].freq if tones else 0.0
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

    # Save PSDs for J(t) and φ(t)  → zároveň si necháme dáta pre PNG
    fJ, PJ = save_psd_csv(t, Jt, str(outdir / "psd_J.csv"))
    fP, PP = save_psd_csv(t, phi_t, str(outdir / "psd_phi.csv"))

    # Save order parameter r(t)
    save_order_parameter_csv(t, res["r"], str(outdir / "order_parameter.csv"))

    # Raw CSV (ak chceš)
    if save_raw:
        import numpy as np
        np.savetxt(outdir / "raw_Jt.csv",
                   np.column_stack([t, Jt]),
                   delimiter=",", header="t_s,J(t)", comments="")
        np.savetxt(outdir / "raw_phi.csv",
                   np.column_stack([t, phi_t]),
                   delimiter=",", header="t_s,phi(t)", comments="")
        # theta snapshot
        np.savetxt(outdir / "theta_final.csv",
                   res["theta"],
                   delimiter=",", header="theta_rad", comments="")

    # Metadata (plne serializovateľné)
    meta = {
        "cfg": asdict(cfg),
        "tones": [asdict(tn) for tn in cfg.drive.tones],
        "notes": "PAREFT scaffold v0.1",
    }
    save_metadata_json(meta, str(outdir / "metadata.json"))

    # PNG počas simulácie (resp. po uložení CSV)
    if save_png:
        _save_plots(outdir, (fJ, PJ), (fP, PP), t, res["r"])

    return {"t": t, "Jt": Jt, "phi_t": phi_t, "r": res["r"], "fJ": fJ, "PJ": PJ, "fP": fP, "PP": PP}
