#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from pareft.utils import analytic_phase, plv


def main():
    ap = argparse.ArgumentParser(description="PAREFT analysis: PSDs and phase locking")
    ap.add_argument("outdir", type=Path, help="Directory with CSV outputs")
    ap.add_argument("--show", action="store_true", help="Show plots interactively")
    args = ap.parse_args()

    psd_J = np.loadtxt(args.outdir/"psd_J.csv", delimiter=",", skiprows=1)
    psd_phi = np.loadtxt(args.outdir/"psd_phi.csv", delimiter=",", skiprows=1)
    r = np.loadtxt(args.outdir/"order_parameter.csv", delimiter=",", skiprows=1)

    # Phase locking between J(t) and φ(t) (proxy)
    # Reconstruct time series for φ(t) and J(t) amplitudes from PSDs is not unique,
    # so here we simply load raw signals if available (optional extension).
    # For now, compute PLV using order parameter phase via Hilbert.
    t = r[:, 0]
    r_t = r[:, 1]

    # If raw signals were saved separately, compute analytic phases; otherwise use r(t).
    # (Extend later to read raw time series files.)
    phi_r = analytic_phase(r_t)
    # Dummy comparison: r vs itself → PLV=1; placeholder for cross-platform use.
    plv_rr = plv(phi_r, phi_r)

    print(json.dumps({
        "PLV_r_r": float(plv_rr),
        "peak_freq_J_Hz": float(psd_J[np.argmax(psd_J[:,1])][0]),
        "peak_freq_phi_Hz": float(psd_phi[np.argmax(psd_phi[:,1])][0]),
    }, indent=2))

    # Plots
    fig1, ax1 = plt.subplots()
    ax1.loglog(psd_J[:,0], psd_J[:,1])
    ax1.set_xlabel("f [Hz]")
    ax1.set_ylabel("PSD[J]")
    ax1.set_title("Drive PSD")

    fig2, ax2 = plt.subplots()
    ax2.loglog(psd_phi[:,0], psd_phi[:,1])
    ax2.set_xlabel("f [Hz]")
    ax2.set_ylabel("PSD[phi]")
    ax2.set_title("Induced Field PSD (proxy)")

    fig3, ax3 = plt.subplots()
    ax3.plot(t, r_t)
    ax3.set_xlabel("t [s]")
    ax3.set_ylabel("r(t)")
    ax3.set_title("Order parameter")

    for p in (fig1, fig2, fig3):
        p.tight_layout()
    if args.show:
        plt.show()
    else:
        for i, p in enumerate((fig1, fig2, fig3), start=1):
            p.savefig(args.outdir/f"plot_{i}.png", dpi=150)

if __name__ == "__main__":
    main()
