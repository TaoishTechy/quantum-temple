import numpy as np

class PhaseEC:
    """
    Lightweight error correction for phase drift:
    - Median filter on phases
    - Outlier clamp
    - Optional ring-wise re-centering
    """
    def __init__(self, k=3, clamp=np.pi/2):
        self.k=k; self.clamp=clamp

    def denoise(self, phases):
        phased = np.asarray(phases, dtype=float)
        # rolling median with padding
        pad = self.k//2
        ext = np.pad(phased, (pad,pad), mode="wrap")
        out = np.empty_like(phased)
        for i in range(len(phased)):
            win = ext[i:i+self.k]
            out[i] = np.median(win)
        # clamp outliers
        delta = out - phased
        delta = np.clip(delta, -self.clamp, self.clamp)
        return phased + delta
