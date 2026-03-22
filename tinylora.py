"""
TinyLoRA: Learning to Reason in 13 Parameters
Paper: https://arxiv.org/abs/2602.04118v1

Core implementation of TinyLoRA:
Essentially an ultra-low-rank adapter that scales LoRA
down to as few as 1 trainable parameter by projecting a shared trainable vector
through fixed random matrices onto truncated SVD directions of frozen weights.
"""

import math
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyLoRALayer(nn.Module):
    """
    Replaces a single frozen nn.Linear with a TinyLoRA-adapted version.

    For a frozen weight W ∈ ℝ^{d_out × d_in}, the update is:

        ΔW = U @ Σ @ R(v) @ Vᵀ

    where U, Σ, V come from the rank-r truncated SVD of W, and:

        R(v) = Σᵢ vᵢ · Pᵢ      (P ∈ ℝ^{u × r × r} are fixed random matrices)

    The trainable vector v ∈ ℝ^u can be shared across modules (weight tying).
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 2,
        proj_dim: int = 1,
        shared_v: Optional[nn.Parameter] = None,
        random_seed: int = 42,
    ):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.proj_dim = proj_dim

        # ── Preserve original dtype (e.g. bfloat16) ──
        orig_dtype = original_linear.weight.data.dtype

        # ── Freeze original weight ──
        W = original_linear.weight.data.float()  # SVD requires float32

        # ── Truncated SVD ──
        U_full, S_full, Vh_full = torch.linalg.svd(W, full_matrices=False)
        U_r = U_full[:, :rank]  # (d_out, r)
        S_r = torch.diag(S_full[:rank])  # (r, r)
        V_r = Vh_full[:rank, :].T  # (d_in, r)

        # Cast back to original dtype for all buffers
        self.register_buffer("U", U_r.to(orig_dtype))  # (d_out, r)
        self.register_buffer("S", S_r.to(orig_dtype))  # (r, r)
        self.register_buffer("V", V_r.to(orig_dtype))  # (d_in, r)

        # ── Frozen original weight (for forward pass) ──
        self.register_buffer("W_frozen", original_linear.weight.data.clone())

        # ── Bias ──
        if original_linear.bias is not None:
            self.register_buffer("bias", original_linear.bias.data.clone())
        else:
            self.bias = None

        # ── Fixed random projection tensor P ∈ ℝ^{u × r × r} ──
        gen = torch.Generator().manual_seed(random_seed)
        P = torch.randn(proj_dim, rank, rank, generator=gen) / math.sqrt(rank)
        # Ensure P is on the same device as the weights
        self.register_buffer("P", P.to(device=W.device, dtype=orig_dtype))

        # ── Trainable vector v ∈ ℝ^u ──
        if shared_v is not None:
            # Weight tying: use externally-owned parameter
            self.v = shared_v
        else:
            self.v = nn.Parameter(torch.zeros(proj_dim, device=W.device))

    def _compute_R(self) -> torch.Tensor:
        """Compute R(v) = Σᵢ vᵢ · Pᵢ  →  (r, r)"""
        # v: (u,)   P: (u, r, r)  →  einsum → (r, r)
        # Ensure v is on the same device as P for calculation
        v = self.v.to(device=self.P.device, dtype=self.P.dtype)
        return torch.einsum("i,ijk->jk", v, self.P)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base frozen output
        base = F.linear(x, self.W_frozen, self.bias)

        # TinyLoRA delta:  x @ V @ Rᵀ @ Σ @ Uᵀ  (transposed because F.linear uses Wᵀ)
        R = self._compute_R()  # (r, r)
        # x: (..., d_in)
        h = x @ self.V  # (..., r)
        # Ensure R is on the same device as h
        h = h @ R.to(h.device).T  # (..., r)
        h = h @ self.S  # (..., r)
        h = h @ self.U.T  # (..., d_out)

        return base + h


# ─── Target module names for common architectures ───────────────────────────

# Covers LLaMA, Qwen, Mistral and similar architectures
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",  # Attention
    "gate_proj",
    "up_proj",
    "down_proj",  # MLP
]


class TinyLoRAModel(nn.Module):
    """
    Wraps a HuggingFace model with TinyLoRA adapters on all target linear layers.

    Args:
        model: A HuggingFace causal LM (e.g. AutoModelForCausalLM).
        rank: Frozen SVD rank r (paper default: 2).
        proj_dim: Dimension u of the trainable vector per group (paper default: 1).
        n_tie: Weight tying factor — number of modules sharing one v vector.
               Set to total number of adapted modules for full tying (= 1 shared v).
               Set to 1 for no tying (each module gets its own v).
        target_modules: List of module name suffixes to adapt.
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int = 2,
        proj_dim: int = 1,
        n_tie: int = 1,
        target_modules: list[str] | None = None,
    ):
        super().__init__()
        self.model = model
        self.rank = rank
        self.proj_dim = proj_dim
        self.n_tie = n_tie
        self.target_modules = target_modules or DEFAULT_TARGET_MODULES

        # Freeze base model
        for param in self.model.parameters():
            param.requires_grad = False

        # Collect target modules
        self._adapted_names: list[str] = []
        targets = self._find_target_modules()

        # Create shared v parameters
        n_groups = max(1, len(targets) // n_tie)
        self.shared_vs = nn.ParameterList(
            [nn.Parameter(torch.zeros(proj_dim)) for _ in range(n_groups)]
        )

        # Replace each target linear with a TinyLoRALayer
        for idx, (name, module) in enumerate(targets):
            group_idx = min(idx // n_tie, n_groups - 1)
            shared_v = self.shared_vs[group_idx]
            layer = TinyLoRALayer(
                original_linear=module,
                rank=rank,
                proj_dim=proj_dim,
                shared_v=shared_v,
                random_seed=42 + idx,  # unique P per module
            )
            self._replace_module(name, layer)
            self._adapted_names.append(name)

        # Report
        total_params = sum(p.numel() for p in self.trainable_parameters())
        print(
            f"[TinyLoRA] Adapted {len(self._adapted_names)} modules, "
            f"{len(self.shared_vs)} shared v groups, "
            f"{total_params} trainable parameters "
            f"({total_params * 2} bytes in bf16)"
        )

    def _find_target_modules(self) -> list[tuple[str, nn.Linear]]:
        """Find all nn.Linear modules matching target names."""
        targets = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if any(name.endswith(t) for t in self.target_modules):
                    targets.append((name, module))
        return targets

    def _replace_module(self, name: str, new_module: nn.Module):
        """Replace a module in the model hierarchy by dotted name."""
        parts = name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    def trainable_parameters(self):
        """Yield only the trainable TinyLoRA parameters."""
        for p in self.shared_vs.parameters():
            yield p

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    # ── Merge / unmerge for inference ────────────────────────────────────────

    def merge(self):
        """Merge TinyLoRA deltas into the frozen weights (for fast inference)."""
        for name in self._adapted_names:
            parts = name.split(".")
            parent = self.model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            layer: TinyLoRALayer = getattr(parent, parts[-1])

            # Ensure everything is on the same device as the target weight
            target_device = layer.W_frozen.device
            R = layer._compute_R().to(target_device)
            U = layer.U.to(target_device)
            S = layer.S.to(target_device)
            V = layer.V.to(target_device)

            delta = U @ S @ R @ V.T  # (d_out, d_in)
            layer.W_frozen.add_(delta)

    def unmerge(self):
        """Remove TinyLoRA deltas from the frozen weights."""
        for name in self._adapted_names:
            parts = name.split(".")
            parent = self.model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            layer: TinyLoRALayer = getattr(parent, parts[-1])

            target_device = layer.W_frozen.device
            R = layer._compute_R().to(target_device)
            U = layer.U.to(target_device)
            S = layer.S.to(target_device)
            V = layer.V.to(target_device)

            delta = U @ S @ R @ V.T
            layer.W_frozen.sub_(delta)

    def get_merged_state_dict(self) -> dict[str, torch.Tensor]:
        """
        Return a state dict with TinyLoRA deltas merged into frozen weights.

        Unlike merge(), this does NOT modify the model in-place. It returns a
        new dict suitable for loading into a separate model (e.g. vLLM).
        """
        # Start from the base model's state dict (shallow copy of tensors)
        sd = {k: v.clone() for k, v in self.model.state_dict().items()}

        for name in self._adapted_names:
            parts = name.split(".")
            parent = self.model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            layer: TinyLoRALayer = getattr(parent, parts[-1])

            R = layer._compute_R()
            delta = layer.U @ layer.S @ R @ layer.V.T  # (d_out, d_in)

            # The key in the state dict for the frozen weight
            weight_key = name + ".W_frozen"
            if weight_key in sd:
                sd[weight_key] = sd[weight_key] + delta.to(sd[weight_key].dtype)
            else:
                # Fallback: try the standard .weight key
                weight_key = name + ".weight"
                if weight_key in sd:
                    sd[weight_key] = sd[weight_key] + delta.to(sd[weight_key].dtype)

        return sd

    # ── Save / load adapter ─────────────────────────────────────────────────

    def save_adapter(self, path: str):
        """Save only the tiny trainable state (v vectors + config)."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(
            {f"v_{i}": v.data for i, v in enumerate(self.shared_vs)},
            p / "adapter.pt",
        )
        config = {
            "rank": self.rank,
            "proj_dim": self.proj_dim,
            "n_tie": self.n_tie,
            "target_modules": self.target_modules,
            "num_adapted": len(self._adapted_names),
            "num_groups": len(self.shared_vs),
            "total_params": self.num_trainable_parameters(),
            "total_bytes_bf16": self.num_trainable_parameters() * 2,
        }
        with open(p / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        print(
            f"[TinyLoRA] Saved adapter to {p}  "
            f"({self.num_trainable_parameters()} params, "
            f"{self.num_trainable_parameters() * 2} bytes)"
        )

    def load_adapter(self, path: str):
        """Load trained v vectors from disk."""
        p = Path(path)
        state = torch.load(p / "adapter.pt", map_location="cpu", weights_only=True)
        for i, v in enumerate(self.shared_vs):
            v.data.copy_(state[f"v_{i}"])
        print(f"[TinyLoRA] Loaded adapter from {p}")
