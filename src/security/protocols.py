# src/security/protocols.py
from __future__ import annotations
import numpy as np
import hashlib

def byzantine_outliers(phases: np.ndarray, z_thresh: float = 3.5) -> np.ndarray:
    """
    Robust outlier detection (MAD-based). Returns boolean mask of suspected nodes.
    """
    x = phases
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    z = 0.6745 * (x - med) / mad
    return np.abs(z) > z_thresh

def phase_mask(phases: np.ndarray, key: str) -> np.ndarray:
    """
    Additive mask (mod 2π) derived from SHA-256(key). Reversible by subtracting same mask.
    """
    h = hashlib.sha256(key.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
    mask = rng.uniform(-np.pi, np.pi, size=phases.size)
    return (phases + mask + np.pi) % (2*np.pi) - np.pi

def verify_integrity(phases: np.ndarray, checksum: str) -> bool:
    """Simple checksum integrity over quantized phases (no crypto, just detection)."""
    q = np.round((phases + np.pi) * 1e3).astype(np.int64)
    dig = hashlib.sha256(q.tobytes()).hexdigest()
    return dig == checksum

def checksum(phases: np.ndarray) -> str:
    q = np.round((phases + np.pi) * 1e3).astype(np.int64)
    return hashlib.sha256(q.tobytes()).hexdigest()
