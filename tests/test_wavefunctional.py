import numpy as np
from quantum_temple_multiverse.mathematics.wavefunctional import WavefunctionalSampler

def test_wavefunctional_sampler_shapes_and_norms():
    wf = WavefunctionalSampler(Dg=4, Df=16, Dc=8, seed=123)
    out = wf.sample(80)
    assert out["g"].shape == (4,)
    assert out["phi"].shape == (16,)
    assert out["psi"].shape == (8,)
    assert abs(np.linalg.norm(out["psi"]) - 1.0) < 1e-6
    assert np.isfinite(out["action"])
    assert out["weight"] >= 0.0
