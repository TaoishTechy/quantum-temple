import numpy as np
from src.core.qudit_state import QuditState
from src.core.resonance_node import ResonanceSync, ResonanceParams
from src.topology.ghost_mesh import GhostMesh

def test_resonance_reduces_variance():
    st = QuditState(128, seed=0)
    mesh = GhostMesh(128)
    sync = ResonanceSync(mesh.neighbors())
    p = ResonanceParams(dt=0.05, k_couple=0.5)
    var0 = st.variance()
    for _ in range(200):
        sync.step(st, p)
    assert st.variance() < var0
