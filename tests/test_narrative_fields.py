from quantum_temple_multiverse.mathematics.narrative_fields import narrative_potential

def test_narrative_potential_monotonicity():
    phi1 = narrative_potential(1, 1, 1, 1)
    phi2 = narrative_potential(2, 1, 1, 1)
    assert phi2 > phi1
