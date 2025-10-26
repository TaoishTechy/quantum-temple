class SigmaQPID:
    def __init__(self, Kp=0.20, Ki=0.05, Kd=0.10, target_var=0.05, init_sigma=0.09):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.target, self.int, self.prev = target_var, 0.0, None
        self.sigma = init_sigma

    def step(self, measured_var: float, dt: float) -> float:
        err = self.target - measured_var
        self.int += err * dt
        der = 0.0 if self.prev is None else (err - self.prev)/dt
        self.prev = err
        self.sigma = max(0.0, self.sigma + self.Kp*err + self.Ki*self.int + self.Kd*der)
        return self.sigma

class PiLock:
    def __init__(self, budget=2): self.budget = budget; self.count = 0
    def try_flip(self, coherence: float, lo=0.55, hi=0.80) -> bool:
        if self.count >= self.budget: return False
        if lo <= coherence <= hi:
            self.count += 1
            return True
        return False

def rla_anchor(ci_target: float = 0.98, alpha: float = 0.12) -> dict:
    return {"ci_target": ci_target, "alpha": alpha}
