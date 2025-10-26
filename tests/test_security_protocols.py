import numpy as np
from src.security.protocols import byzantine_outliers, phase_mask, checksum, verify_integrity

def test_outlier_detection():
    x = np.zeros(100); x[3] = 3.14
    bad = byzantine_outliers(x, z_thresh=3.0)
    assert bad.sum() >= 1

def test_mask_and_verify():
    x = np.linspace(-1,1,50)
    cs = checksum(x)
    assert verify_integrity(x, cs)
    y = phase_mask(x, key="secret")
    assert not verify_integrity(y, cs)
