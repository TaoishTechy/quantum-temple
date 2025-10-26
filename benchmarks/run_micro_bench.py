# benchmarks/run_micro_bench.py
import json, time, pathlib
import numpy as np
from src.core.qudit_state import QuditState
from src.core.resonance_node import ResonanceSync, ResonanceParams
from src.topology.ghost_mesh import GhostMesh
from src.operators.symbolic_operators import (
    nabla_zeta, nabla_P_oracle, nabla_C_collatz, nabla_T_gap_closure
)

def run(n=512, steps=5000, seed=0):
    mesh = GhostMesh(n, cross_links=[(0, n//2), (n//4, 3*n//4)])
    nb = mesh.neighbors()
    sync = ResonanceSync(nb)
    params = ResonanceParams(dt=0.02, k_couple=0.35, noise_std=0.0)

    st = QuditState(n=n, seed=seed)
    out = []
    t0 = time.time()
    for k in range(steps):
        # symbolic pipeline
        if k % 7 == 0:  nabla_zeta(st, eta=0.01)
        if k % 11 == 0: nabla_P_oracle(st, clauses_sat=0.6, gain=0.02)
        if k % 13 == 0: nabla_C_collatz(st, cycles=1, coeff=0.015)
        if k % 17 == 0: nabla_T_gap_closure(st, alpha=0.10)

        # resonance step
        sync.step(st, params)

        if k % 50 == 0:
            out.append({
                "step": k,
                "plv": st.plv(),
                "var": st.variance()
            })
    dt = time.time() - t0
    return out, dt

if __name__ == "__main__":
    logs, elapsed = run()
    path = pathlib.Path("data/benchmarks/cpu_micro.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in logs:
            f.write(json.dumps(row) + "\n")
    print(f"✅ wrote {path}  | elapsed={elapsed:.2f}s")
