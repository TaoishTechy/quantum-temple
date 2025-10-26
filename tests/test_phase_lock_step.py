import numpy as np
from src.core.state import QuditNode, ResonanceState
from src.core.operators import H_stab

def test_h_stab_moves_to_mean():
    nodes = [QuditNode(phase=p) for p in (-1.0, 0.0, 1.0)]
    s = ResonanceState(nodes=nodes)
    before = np.array([n.phase for n in nodes])
    H_stab(s, g=1.0)
    after = np.array([n.phase for n in nodes])
    # Everyone should move ~toward circular mean
    assert np.var(after) < np.var(before)
