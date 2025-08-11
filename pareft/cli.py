#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from pareft.config import load_config, Tone
from pareft.simulate import run_experiment

CHANNELS = {
    "C1": {"amp_scale": 0.8}, "C2": {"amp_scale": 0.9}, "C3": {"amp_scale": 1.0},
    "C4": {"amp_scale": 1.2}, "C5": {"amp_scale": 1.0}, "C6": {"amp_scale": 1.1}
}

def main():
    ap = argparse.ArgumentParser(description="PAREFT CLI runner")
    ap.add_argument("config", type=Path, help="YAML config template")
    ap.add_argument("--triplet", type=str, default="C2,C3,C6",
                    help="Comma-separated control channels, e.g., C2,C3,C6")
    ap.add_argument("--durations", type=str, default="10,40,10",
                    help="Comma-separated durations in seconds for the triplet")
    ap.add_argument("--freqs", type=str, default="1.0,3.0,5.0",
                    help="Comma-separated frequencies (Hz)")
    ap.add_argument("--amps", type=str, default="0.4,0.8,0.6",
                    help="Comma-separated amplitudes")
    ap.add_argument("--phases", type=str, default="0.0,1.57,0.0",
                    help="Comma-separated phases (rad)")
    ap.add_argument("--save-raw", type=int, choices=[0,1], default=0,
                    help="Save raw time series CSV (0/1)")
    args = ap.parse_args()

    cfg = load_config(args.config)

    ch = [c.strip().upper() for c in args.triplet.split(",")]
    dur = [float(x) for x in args.durations.split(",")]
    freqs = [float(x) for x in args.freqs.split(",")]
    amps = [float(x) for x in args.amps.split(",")]
    phases = [float(x) for x in args.phases.split(",")]

    if not (len(ch) == len(dur) == len(freqs) == len(amps) == len(phases)):
        raise SystemExit("All lists must have the same length")

    tones = []
    for i in range(len(ch)):
        scale = CHANNELS.get(ch[i], {"amp_scale": 1.0})["amp_scale"]
        tones.append(Tone(freq=freqs[i], amp=amps[i]*scale, phase=phases[i]))

    # Override drive spec
    cfg.drive.tones = tones
    cfg.drive.duration = sum(dur)

    run_experiment(cfg, save_raw=bool(args.save_raw), save_png=True)
    print("Finished. Outputs in", cfg.outdir)

if __name__ == "__main__":
    main()
