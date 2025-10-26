import numpy as np
from src.core.qudit_state import QuditState

def test_plv_bounds_and_determinism():
    a = QuditState(64, seed=1); b = QuditState(64, seed=1)
    assert abs(a.plv()) <= 1.0
    assert np.allclose(a.phases, b.phases)

def test_phase_add_wrap():
    st = QuditState(8, seed=0)
    st.add_phase(10*np.pi)
    assert np.all(st.phases <= np.pi) and np.all(st.phases >= -np.pi)
