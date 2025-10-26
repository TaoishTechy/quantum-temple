import torch

class TorchBridge:
    """
    Move resonance state into Torch tensors for hybrid ops or training.
    """
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def phases_to_tensor(self, phases):
        t = torch.tensor(phases, dtype=torch.float32, device=self.device)
        return t

    def update_from_tensor(self, state, t):
        vals = t.detach().to("cpu").numpy().tolist()
        for n, v in zip(state.nodes, vals):
            n.phase = float(v)
