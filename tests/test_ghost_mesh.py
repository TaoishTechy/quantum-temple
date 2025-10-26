from src.topology.ghost_mesh import GhostMesh

def test_neighbors_sane():
    g = GhostMesh(10, cross_links=[(0,5)])
    nb = g.neighbors()
    assert set(nb[0]) >= {9,1,5}
    assert set(nb[5]) >= {4,6,0}
