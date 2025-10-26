import time, numpy as np
from quantum_temple_multiverse.multiverse.core import QuantumMultiverse

if __name__ == "__main__":
    qm = QuantumMultiverse()
    rid = qm.spawn_reality(
        {"seed":1,"m_story":2,"m_meaning":3,"r_context":4,"G_narrative":1.0},
        {"c":1.0,"G":1.0,"hbar":1.0},
        {"K":32}
    )
    t0 = time.time()
    N = 25
    for _ in range(N):
        qm.tick(rid, dt=1.0)
    dt = time.time() - t0
    print(f"ticks={N} time={dt:.3f}s per_tick={dt/N:.4f}s")
