from __future__ import annotations
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict

@dataclass
class Tone:
    freq: float  # Hz
    amp: float   # arbitrary units of drive
    phase: float # radians

@dataclass
class Envelope:
    kind: str = "hann"  # "hann", "rect", "exp"
    tau: float = 1.0    # s, ramp time where relevant

@dataclass
class SpatialMask:
    kind: str = "uniform"  # "uniform", "gaussian", "ring"
    sigma: float = 1.0     # mask parameter (units arbitrary here)

@dataclass
class DriveConfig:
    tones: List[Tone]
    envelope: Envelope
    mask: SpatialMask
    duration: float
    dt: float

@dataclass
class NetworkConfig:
    N: int
    K: float
    M: float
    gamma: float
    omega_spread: float
    delay: float = 0.0
    noise_std: float = 0.0

@dataclass
class EFTConfig:
    Lambda: float
    c_sigma: float
    c_a: float
    m_phi: float
    lambda_phi: float
    damping: float

@dataclass
class ExperimentConfig:
    drive: DriveConfig
    network: NetworkConfig
    eft: EFTConfig
    seed: int = 1234
    outdir: str = "outputs"


def load_config(path: str | Path) -> ExperimentConfig:
    data = yaml.safe_load(Path(path).read_text())
    tones = [Tone(**t) for t in data["drive"]["tones"]]
    env = Envelope(**data["drive"].get("envelope", {}))
    mask = SpatialMask(**data["drive"].get("mask", {}))
    drive = DriveConfig(
        tones=tones,
        envelope=env,
        mask=mask,
        duration=float(data["drive"]["duration"]),
        dt=float(data["drive"]["dt"]),
    )
    net = NetworkConfig(**data["network"])
    eft = EFTConfig(**data["eft"])
    return ExperimentConfig(drive=drive, network=net, eft=eft, seed=int(data.get("seed", 1234)), outdir=data.get("outdir", "outputs"))
