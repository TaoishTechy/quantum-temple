engine:
  dt: 0.001
  steps: 1_000_000
  alpha: 0.12        # forward damping
  sigma_q:
    mode: adaptive
    initial: 0.09
    target_variance: 0.05
    pid: { Kp: 0.20, Ki: 0.05, Kd: 0.10 }
retro_causal_anchor:
  mode: soft_exponential
ledger:
  path: data/governance_ledger.csv
parity_lock:
  budget: 2
