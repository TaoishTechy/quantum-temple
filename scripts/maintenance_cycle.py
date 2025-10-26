"""
Maintenance cycle: Monitor-only (R_meta + Δ_purity), log metrics snapshot.
No creation; sustain ring coherence. CPU-only and memory-lean.
"""
from __future__ import annotations
import time, json, pathlib
from src.core.qudit_state import QuditState
from src.core.resonance_node import ResonanceSync, ResonanceParams
from src.topology.ghost_mesh import GhostMesh
from src.operators.maintenance import r_meta_432hz, delta_purity

def run_once(n=2401, seed=1111, steps=240, dt=1/240.0, out="data/snapshots/maintenance.json"):
    mesh = GhostMesh(n, cross_links=[(0, n//2), (n//4, 3*n//4)])
    sync = ResonanceSync(mesh.neighbors())
    params = ResonanceParams(dt=dt, k_couple=0.0)  # no coupling changes; monitor-only
    st = QuditState(n=n, seed=seed)

    t0 = time.time()
    t = 0.0
    for _ in range(steps):
        r_meta_432hz(st, t, gain=1.0)
        delta_purity(st, eta=0.003)
        sync.step(st, params)  # dt applied; k_couple=0 keeps topology idle
        t += dt

    # Write snapshot
    path = pathlib.Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    snap = {
        "nodes": n, "seed": seed, "dt": dt, "steps": steps,
        "metrics": {
            "PLV_total": st.plv(),
            "variance": st.variance(),
            "purity_proxy": 1.0 - min(1.0, st.variance()), # simple monotone proxy
        },
        "ts": time.time() - t0
    }
    path.write_text(json.dumps(snap, indent=2))
    print(f"✅ Maintenance snapshot written → {path}")

if __name__ == "__main__":
    run_once()
