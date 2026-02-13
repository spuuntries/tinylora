"""
Unit tests for TinyLoRA core module.
Run with: python test_tinylora.py
"""

import torch
import torch.nn as nn
from tinylora import TinyLoRALayer, TinyLoRAModel


def test_layer_shapes():
    """TinyLoRALayer should produce the same output shape as the original linear."""
    print("test_layer_shapes...", end=" ")
    d_in, d_out = 64, 128
    linear = nn.Linear(d_in, d_out, bias=True)
    layer = TinyLoRALayer(linear, rank=2, proj_dim=3)

    x = torch.randn(2, 10, d_in)
    y = layer(x)
    assert y.shape == (2, 10, d_out), f"Expected (2, 10, {d_out}), got {y.shape}"
    print("PASSED")


def test_layer_no_bias():
    """TinyLoRALayer should work without bias."""
    print("test_layer_no_bias...", end=" ")
    d_in, d_out = 32, 64
    linear = nn.Linear(d_in, d_out, bias=False)
    layer = TinyLoRALayer(linear, rank=1, proj_dim=2)

    x = torch.randn(4, d_in)
    y = layer(x)
    assert y.shape == (4, d_out)
    print("PASSED")


def test_zero_init():
    """With v=0, TinyLoRA output should equal the original linear output."""
    print("test_zero_init...", end=" ")
    d_in, d_out = 32, 64
    linear = nn.Linear(d_in, d_out)
    layer = TinyLoRALayer(linear, rank=2, proj_dim=1)

    x = torch.randn(3, d_in)
    y_original = linear(x)
    y_tinylora = layer(x)

    assert torch.allclose(y_original, y_tinylora, atol=1e-4), \
        f"Max diff: {(y_original - y_tinylora).abs().max().item()}"
    print("PASSED")


def test_weight_tying():
    """Modules with shared v should have the same trainable parameter."""
    print("test_weight_tying...", end=" ")
    shared_v = nn.Parameter(torch.zeros(2))
    l1 = TinyLoRALayer(nn.Linear(32, 64), rank=2, proj_dim=2, shared_v=shared_v)
    l2 = TinyLoRALayer(nn.Linear(32, 64), rank=2, proj_dim=2, shared_v=shared_v)

    assert l1.v is l2.v, "v should be the same object"
    assert l1.v.data_ptr() == l2.v.data_ptr(), "v should share memory"
    print("PASSED")


def test_parameter_count():
    """TinyLoRALayer with proj_dim=u should have exactly u trainable params (when not shared)."""
    print("test_parameter_count...", end=" ")
    u = 5
    layer = TinyLoRALayer(nn.Linear(32, 64), rank=2, proj_dim=u)
    trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    assert trainable == u, f"Expected {u} trainable params, got {trainable}"
    print("PASSED")


def test_nonzero_delta():
    """After modifying v, the output should differ from the frozen output."""
    print("test_nonzero_delta...", end=" ")
    linear = nn.Linear(32, 64)
    layer = TinyLoRALayer(linear, rank=2, proj_dim=1)

    x = torch.randn(3, 32)
    y_before = layer(x).clone()

    # Modify v
    with torch.no_grad():
        layer.v.fill_(1.0)

    y_after = layer(x)
    assert not torch.allclose(y_before, y_after, atol=1e-6), \
        "Output should change when v is modified"
    print("PASSED")


def test_merge_correctness():
    """After merging, forward pass through W_frozen alone should match the adapted output."""
    print("test_merge_correctness...", end=" ")
    d_in, d_out = 32, 64
    linear = nn.Linear(d_in, d_out, bias=False)
    layer = TinyLoRALayer(linear, rank=2, proj_dim=2)

    with torch.no_grad():
        layer.v.fill_(0.5)

    x = torch.randn(3, d_in)
    y_adapted = layer(x).clone()

    # Merge delta into W_frozen
    R = layer._compute_R()
    delta = layer.U @ layer.S @ R @ layer.V.T
    layer.W_frozen.add_(delta)

    # Now zero out v so delta contribution in forward is zero
    with torch.no_grad():
        layer.v.zero_()

    y_merged = layer(x)
    assert torch.allclose(y_adapted, y_merged, atol=1e-4), \
        f"Max diff: {(y_adapted - y_merged).abs().max().item()}"
    print("PASSED")


def test_model_wrapper():
    """TinyLoRAModel should correctly wrap a simple transformer-like model."""
    print("test_model_wrapper...", end=" ")

    # Create a toy model that looks like a transformer layer
    class ToyTransformer(nn.Module):
        def __init__(self, d=64):
            super().__init__()
            self.q_proj = nn.Linear(d, d)
            self.k_proj = nn.Linear(d, d)
            self.v_proj = nn.Linear(d, d)
            self.o_proj = nn.Linear(d, d)
            self.gate_proj = nn.Linear(d, d * 4)
            self.up_proj = nn.Linear(d, d * 4)
            self.down_proj = nn.Linear(d * 4, d)

        def forward(self, x):
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            o = self.o_proj(q + k + v)
            g = torch.sigmoid(self.gate_proj(o))
            u = self.up_proj(o)
            return self.down_proj(g * u)

    toy = ToyTransformer(d=64)
    wrapped = TinyLoRAModel(toy, rank=2, proj_dim=1, n_tie=7)

    n_params = wrapped.num_trainable_parameters()
    print(f"({n_params} params)", end=" ")
    assert n_params > 0, "Should have trainable parameters"
    assert n_params <= 7, f"With n_tie=7 and proj_dim=1, expected ≤7 params, got {n_params}"

    x = torch.randn(2, 5, 64)
    y = wrapped(x)
    assert y.shape == (2, 5, 64), f"Expected (2, 5, 64), got {y.shape}"
    print("PASSED")


def test_save_load(tmp_path="./test_adapter_tmp"):
    """save_adapter and load_adapter should round-trip correctly."""
    print("test_save_load...", end=" ")
    import shutil
    from pathlib import Path

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(32, 32)
            self.k_proj = nn.Linear(32, 32)
        def forward(self, x):
            return self.q_proj(x) + self.k_proj(x)

    model = TinyModel()
    wrapped = TinyLoRAModel(model, rank=1, proj_dim=2, n_tie=1,
                            target_modules=["q_proj", "k_proj"])

    # Set some values
    with torch.no_grad():
        for v in wrapped.shared_vs:
            v.fill_(0.42)

    # Save
    wrapped.save_adapter(tmp_path)

    # Create fresh model and load
    model2 = TinyModel()
    wrapped2 = TinyLoRAModel(model2, rank=1, proj_dim=2, n_tie=1,
                             target_modules=["q_proj", "k_proj"])
    wrapped2.load_adapter(tmp_path)

    for v1, v2 in zip(wrapped.shared_vs, wrapped2.shared_vs):
        assert torch.allclose(v1, v2), "Loaded v should match saved v"

    # Cleanup
    shutil.rmtree(Path(tmp_path), ignore_errors=True)
    print("PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("TinyLoRA Unit Tests")
    print("=" * 60)

    test_layer_shapes()
    test_layer_no_bias()
    test_zero_init()
    test_weight_tying()
    test_parameter_count()
    test_nonzero_delta()
    test_merge_correctness()
    test_model_wrapper()
    test_save_load()

    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
