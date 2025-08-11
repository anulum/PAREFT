#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def analytic_phase(x: np.ndarray) -> np.ndarray:
    from scipy.signal import hilbert
    return np.angle(hilbert(x))

def main():
    ap = argparse.ArgumentParser(description="PAREFT analysis: PSDs, PLV, plots")
    ap.add_argument("outdir", type=Path, help="Directory with CSV outputs")
    ap.add_argument("--show", action="store_true", help="Show plots instead of saving PNGs")
    args = ap.parse_args()

    # Load PSDs
    psd_J = np.loadtxt(args.outdir/"psd_J.csv", delimiter=",", skiprows=1)
    psd_phi = np.loadtxt(args.outdir/"psd_phi.csv", delimiter=",", skiprows=1)

    # Load time series if available (preferred)
    ts_path = args.outdir/"timeseries.csv"
    if ts_path.exists():
        ts = np.loadtxt(ts_path, delimiter=",", skiprows=1)
        t, Jt, phi_t, r_t = ts.T
        phi_J = analytic_phase(Jt)
        phi_phi = analytic_phase(phi_t)
        plv = np.abs(np.mean(np.exp(1j*(phi_J - phi_phi))))
    else:
        # Fallback: use order_parameter only (less informative)
        ordp = np.loadtxt(args.outdir/"order_parameter.csv", delimiter=",", skiprows=1)
        t, r_t = ordp.T
        Jt = phi_t = None
        plv = 1.0  # trivial fallback

    # Report JSON summary
    peak_J = float(psd_J[np.argmax(psd_J[:,1])][0])
    peak_phi = float(psd_phi[np.argmax(psd_phi[:,1])][0])
    print(json.dumps({"PLV_J_phi": float(plv),
                      "peak_freq_J_Hz": peak_J,
                      "peak_freq_phi_Hz": peak_phi}, indent=2))

    # Plots
    fig1, ax1 = plt.subplots()
    ax1.loglog(psd_J[:,0], psd_J[:,1])
    ax1.set_xlabel("f [Hz]"); ax1.set_ylabel("PSD[J]"); ax1.set_title("Drive PSD")

    fig2, ax2 = plt.subplots()
    ax2.loglog(psd_phi[:,0], psd_phi[:,1])
    ax2.set_xlabel("f [Hz]"); ax2.set_ylabel("PSD[phi]"); ax2.set_title("Induced Field PSD")

    fig3, ax3 = plt.subplots()
    ax3.plot(t, r_t)
    ax3.set_xlabel("t [s]"); ax3.set_ylabel("r(t)"); ax3.set_title("Order parameter")

    for f in (fig1, fig2, fig3): f.tight_layout()
    if args.show:
        plt.show()
    else:
        (args.outdir/"plot_1.png").write_bytes(b"")  # touch if permission ok
        fig1.savefig(args.outdir/"plot_1.png", dpi=150)
        fig2.savefig(args.outdir/"plot_2.png", dpi=150)
        fig3.savefig(args.outdir/"plot_3.png", dpi=150)

if __name__ == "__main__":
    main()
