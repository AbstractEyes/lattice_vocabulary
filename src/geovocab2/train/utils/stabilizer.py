# ─────────────────────────────────────────────────────────────────────────────
# Pentachora On-Demand Parameter Stabilizer (flat-bound, CUDA-graph optional)
# ─────────────────────────────────────────────────────────────────────────────
import torch, time
from torch import nn

PENTA_STAB_CFG = {
    "alpha": 0.30,          # exp damping strength
    "p_exp": 2.0,           # use 2.0 for (x*x) fast path
    "vel_gain": 1.0,        # velocity influence on clamp
    "base_clamp": 256.0,    # baseline clamp
    "hard_ceiling": 1e6,    # absolute guard
    "on_demand_eps": 1.01,  # trigger threshold multiplier
    "use_cuda_graph": True, # capture/replay kernel sequence on CUDA
}

class _FlatBinder:
    def __init__(self, module: nn.Module):
        params = [p for p in module.parameters() if p.requires_grad and p.is_floating_point()]
        if not params:
            raise ValueError("No float params to bind.")
        self.params = params
        self.dev, self.dt = params[0].device, params[0].dtype
        total = sum(p.numel() for p in params)
        self.flat = torch.empty(total, device=self.dev, dtype=self.dt)
        self.grad = torch.zeros_like(self.flat)
        cur = 0
        with torch.no_grad():
            for p in params:
                n = p.numel()
                self.flat[cur:cur+n].copy_(p.data.view(-1))
                p.data = self.flat[cur:cur+n].view_as(p.data)
                p.grad = self.grad[cur:cur+n].view_as(p.data)
                cur += n

class PentachoraStabilizer:
    def __init__(self, cfg: dict = None):
        self.cfg = {**PENTA_STAB_CFG, **(cfg or {})}
        self.binder = None
        self.graph = None
        self.triggers = 0
        self.last_stats = {}

    def bind(self, module: nn.Module):
        self.binder = _FlatBinder(module)
        if torch.cuda.is_available() and self.cfg["use_cuda_graph"]:
            self._build_cuda_graph()

    @torch.no_grad()
    def _should_gate(self) -> bool:
        w_max = self.binder.flat.abs().max()
        g_max = self.binder.grad.abs().max()
        thresh = self.cfg["base_clamp"] * (1.0 + self.cfg["vel_gain"] * g_max)
        need = bool((w_max > self.cfg["on_demand_eps"] * thresh).item())
        self.last_stats = {"w_max": float(w_max), "g_max": float(g_max), "thresh": float(thresh)}
        return need

    def _build_cuda_graph(self):
        w, g = self.binder.flat, self.binder.grad
        self.graph = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        w.zero_(); g.zero_(); torch.cuda.synchronize()
        with torch.cuda.graph(self.graph, stream=s):
            alpha, pexp = self.cfg["alpha"], self.cfg["p_exp"]
            v_scale = 1.0 + self.cfg["vel_gain"] * g.abs()
            clamp = self.cfg["base_clamp"] * v_scale
            mag = w.abs()
            rel = (mag / (clamp + 1e-8)) - 1.0
            over = torch.relu(rel) if pexp == 2.0 else torch.relu(rel).pow(pexp)
            damp = torch.exp(-alpha * (over if pexp != 2.0 else over * over))
            w.mul_(damp).clamp_(-self.cfg["hard_ceiling"], self.cfg["hard_ceiling"])

    @torch.no_grad()
    def _gate_now(self):
        # CPU path or CUDA without graphs
        w, g = self.binder.flat, self.binder.grad
        alpha, pexp = self.cfg["alpha"], self.cfg["p_exp"]
        v_scale = 1.0 + self.cfg["vel_gain"] * g.abs()
        clamp = self.cfg["base_clamp"] * v_scale
        mag = w.abs()
        rel = (mag / (clamp + 1e-8)) - 1.0
        if pexp == 2.0:
            over = torch.relu(rel); damp = torch.exp(-alpha * (over * over))
        else:
            damp = torch.exp(-alpha * torch.relu(rel).pow(pexp))
        w.mul_(damp).clamp_(-self.cfg["hard_ceiling"], self.cfg["hard_ceiling"])

    @torch.no_grad()
    def step_post_update(self):
        """Call immediately after optimizer.step()."""
        if self.binder is None:
            raise RuntimeError("PentachoraStabilizer.bind(model) must be called once before training.")
        if not self._should_gate():
            return
        self.triggers += 1
        if torch.cuda.is_available() and self.graph is not None:
            self.graph.replay()
        else:
            self._gate_now()
