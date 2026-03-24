from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import importlib
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triton_qwen3vl import TRITON_AVAILABLE, patch
from triton_qwen3vl.backends import text_triton_attention, vision_triton_attention
from triton_qwen3vl.gqa_flash_attention import gqa_flash_attention
from triton_qwen3vl.gqa_flash_attention import _select_gqa_prefill_launch_config
from triton_qwen3vl.fallback import repeat_kv
from triton_qwen3vl.gqa_decode_attention import gqa_decode_attention
from triton_qwen3vl.varlen_flash_attention import _select_varlen_launch_config, varlen_flash_attention


class DummyRegistry(dict):
    def register(self, name, fn):
        self[name] = fn


class DummyAttentionModule(torch.nn.Module):
    def __init__(self, *, num_key_value_groups: int = 1, scaling: float = 0.125, is_causal: bool = True):
        super().__init__()
        self.num_key_value_groups = num_key_value_groups
        self.scaling = scaling
        self.is_causal = is_causal


def _build_fake_model():
    vision_runtime_config = SimpleNamespace(_attn_implementation="sdpa")
    text_runtime_config = SimpleNamespace(_attn_implementation="sdpa")
    return SimpleNamespace(
        model=SimpleNamespace(
            visual=SimpleNamespace(config=vision_runtime_config),
            language_model=SimpleNamespace(config=text_runtime_config),
        ),
        config=SimpleNamespace(
            vision_config=SimpleNamespace(_attn_implementation="sdpa"),
            text_config=SimpleNamespace(_attn_implementation="sdpa"),
        ),
    )


def test_install_qwen3vl_triton_attention_registers_backends_and_mutates_configs(monkeypatch):
    attention_registry = DummyRegistry()
    mask_registry = DummyRegistry({"flash_attention_2": object()})
    monkeypatch.setattr(patch, "_load_attention_registries", lambda: (attention_registry, mask_registry))

    model = _build_fake_model()
    state = patch.install_qwen3vl_triton_attention(model)

    assert patch.VISION_BACKEND_KEY in attention_registry
    assert patch.TEXT_BACKEND_KEY in attention_registry
    assert patch.VISION_BACKEND_KEY in mask_registry
    assert patch.TEXT_BACKEND_KEY in mask_registry
    assert model.model.visual.config._attn_implementation == patch.VISION_BACKEND_KEY
    assert model.model.language_model.config._attn_implementation == patch.TEXT_BACKEND_KEY
    assert model.config.vision_config._attn_implementation == patch.VISION_BACKEND_KEY
    assert model.config.text_config._attn_implementation == patch.TEXT_BACKEND_KEY
    assert state["installed"] is True
    assert "mutated_configs" in state


def test_text_triton_attention_uses_sdpa_fallback_on_cpu(monkeypatch):
    sentinel = torch.randn(1, 4, 2, 8)
    recorded = {}

    def fake_sdpa(*args, **kwargs):
        recorded["called"] = True
        return sentinel, None

    monkeypatch.setattr("triton_qwen3vl.backends.sdpa_fallback", fake_sdpa)

    module = DummyAttentionModule(num_key_value_groups=2, is_causal=True)
    query = torch.randn(1, 4, 4, 8)
    key = torch.randn(1, 2, 4, 8)
    value = torch.randn(1, 2, 4, 8)

    output, attn_weights = text_triton_attention(
        module,
        query,
        key,
        value,
        attention_mask=None,
        dropout=0.0,
        scaling=None,
    )

    assert recorded["called"] is True
    assert attn_weights is None
    assert output is sentinel


def test_text_triton_attention_uses_gqa_kernel_fast_path(monkeypatch):
    sentinel = torch.arange(1 * 4 * 5 * 4, dtype=torch.float32).reshape(1, 4, 5, 4)
    recorded = {}

    monkeypatch.setattr("triton_qwen3vl.backends.can_use_gqa_flash_attention", lambda *args, **kwargs: True)

    def fake_gqa(query, key, value, **kwargs):
        del query, key, value
        recorded.update(kwargs)
        return sentinel

    monkeypatch.setattr("triton_qwen3vl.backends.gqa_flash_attention", fake_gqa)

    def unexpected_fallback(*args, **kwargs):
        raise AssertionError("text fast path should not fall back to sdpa")

    monkeypatch.setattr("triton_qwen3vl.backends.sdpa_fallback", unexpected_fallback)

    module = DummyAttentionModule(num_key_value_groups=2, scaling=0.25, is_causal=True)
    query = torch.randn(1, 4, 5, 4)
    key = torch.randn(1, 2, 5, 4)
    value = torch.randn(1, 2, 5, 4)

    output, attn_weights = text_triton_attention(
        module,
        query,
        key,
        value,
        attention_mask=None,
        dropout=0.0,
        scaling=None,
    )

    assert recorded["sm_scale"] == module.scaling
    assert recorded["num_key_value_groups"] == 2
    assert recorded["causal"] is True
    assert attn_weights is None
    assert torch.equal(output, sentinel.transpose(1, 2).contiguous())


def test_gqa_flash_attention_uses_training_kernel_when_grad_enabled(monkeypatch):
    sentinel = torch.arange(1 * 4 * 5 * 4, dtype=torch.float32).reshape(1, 4, 5, 4)
    recorded = {}
    gqa_module = importlib.import_module("triton_qwen3vl.gqa_flash_attention")

    def fake_training(query, key, value, **kwargs):
        recorded["query_shape"] = tuple(query.shape)
        recorded["key_shape"] = tuple(key.shape)
        recorded["value_shape"] = tuple(value.shape)
        recorded.update(kwargs)
        return sentinel

    monkeypatch.setattr(gqa_module, "dense_flash_attention_training", fake_training)

    query = torch.randn(1, 4, 5, 4, requires_grad=True)
    key = torch.randn(1, 2, 5, 4, requires_grad=True)
    value = torch.randn(1, 2, 5, 4, requires_grad=True)

    output = gqa_flash_attention(
        query,
        key,
        value,
        causal=True,
        sm_scale=0.25,
        num_key_value_groups=2,
    )

    assert recorded["query_shape"] == (1, 4, 5, 4)
    assert recorded["key_shape"] == (1, 4, 5, 4)
    assert recorded["value_shape"] == (1, 4, 5, 4)
    assert recorded["sm_scale"] == 0.25
    assert recorded["causal"] is True
    assert torch.equal(output, sentinel)


def test_select_gqa_prefill_launch_config_uses_stable_defaults_without_experimental_flag(monkeypatch):
    monkeypatch.delenv("TRITON_QWEN3VL_HIGH_PARALLELISM", raising=False)
    query = torch.randn(1, 16, 1024, 128)
    block_m, block_n, num_warps, num_stages = _select_gqa_prefill_launch_config(query)

    assert block_m == 128
    assert block_n == 64
    assert num_warps == 8
    assert num_stages == 2


def test_select_gqa_prefill_launch_config_prefers_more_grid_parallelism_for_small_batches(monkeypatch):
    monkeypatch.setenv("TRITON_QWEN3VL_HIGH_PARALLELISM", "1")
    query = torch.randn(1, 16, 1024, 128)
    block_m, block_n, num_warps, num_stages = _select_gqa_prefill_launch_config(query)

    assert block_m == 64
    assert block_n == 64
    assert num_warps == 4
    assert num_stages == 3


def test_varlen_flash_attention_backward_uses_training_kernel(monkeypatch):
    varlen_module = importlib.import_module("triton_qwen3vl.varlen_flash_attention")
    recorded = {"calls": 0}

    def fake_forward(query, key, value, **kwargs):
        del key, value, kwargs
        return query.clone()

    def fake_training(query, key, value, **kwargs):
        recorded["calls"] += 1
        recorded["causal"] = kwargs["causal"]
        recorded["sm_scale"] = kwargs["sm_scale"]
        return query + key + value

    monkeypatch.setattr(varlen_module, "_varlen_flash_attention_forward", fake_forward)
    monkeypatch.setattr(varlen_module, "dense_flash_attention_training", fake_training)

    query = torch.randn(1, 2, 5, 4, requires_grad=True)
    key = torch.randn(1, 2, 5, 4, requires_grad=True)
    value = torch.randn(1, 2, 5, 4, requires_grad=True)
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)

    output = varlen_flash_attention(
        query,
        key,
        value,
        cu_seq_lens_q=cu,
        cu_seq_lens_k=cu,
        sm_scale=0.25,
        max_length_q=3,
        max_length_k=3,
    )
    output.backward(torch.ones_like(output))

    assert recorded["calls"] == 2
    assert recorded["causal"] is False
    assert recorded["sm_scale"] == 0.25


def test_text_triton_attention_uses_decode_kernel_fast_path(monkeypatch):
    sentinel = torch.arange(1 * 4 * 1 * 4, dtype=torch.float32).reshape(1, 4, 1, 4)
    recorded = {}

    monkeypatch.setattr("triton_qwen3vl.backends.can_use_gqa_decode_attention", lambda *args, **kwargs: True)

    def fake_decode(query, key, value, **kwargs):
        del query, key, value
        recorded.update(kwargs)
        return sentinel

    monkeypatch.setattr("triton_qwen3vl.backends.gqa_decode_attention", fake_decode)

    def unexpected_prefill(*args, **kwargs):
        raise AssertionError("decode fast path should not call prefill kernel")

    def unexpected_fallback(*args, **kwargs):
        raise AssertionError("decode fast path should not fall back to sdpa")

    monkeypatch.setattr("triton_qwen3vl.backends.gqa_flash_attention", unexpected_prefill)
    monkeypatch.setattr("triton_qwen3vl.backends.sdpa_fallback", unexpected_fallback)

    module = DummyAttentionModule(num_key_value_groups=2, scaling=0.25, is_causal=True)
    query = torch.randn(1, 4, 1, 4)
    key = torch.randn(1, 2, 5, 4)
    value = torch.randn(1, 2, 5, 4)

    output, attn_weights = text_triton_attention(
        module,
        query,
        key,
        value,
        attention_mask=None,
        dropout=0.0,
        scaling=None,
    )

    assert recorded["sm_scale"] == module.scaling
    assert recorded["num_key_value_groups"] == 2
    assert attn_weights is None
    assert torch.equal(output, sentinel.transpose(1, 2).contiguous())


def test_vision_triton_attention_varlen_fallback_preserves_packed_layout(monkeypatch):
    calls = []

    def fake_sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs):
        del module, key, value, attention_mask, dropout, scaling, is_causal, kwargs
        calls.append(query.shape[2])
        return query.transpose(1, 2).contiguous() + 1, None

    monkeypatch.setattr("triton_qwen3vl.backends.sdpa_fallback", fake_sdpa)

    module = DummyAttentionModule(num_key_value_groups=1, scaling=0.25, is_causal=False)
    query = torch.randn(1, 2, 5, 4)
    key = torch.randn(1, 2, 5, 4)
    value = torch.randn(1, 2, 5, 4)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)

    output, attn_weights = vision_triton_attention(
        module,
        query,
        key,
        value,
        attention_mask=None,
        dropout=0.0,
        scaling=None,
        cu_seq_lens_q=cu_seqlens,
        cu_seq_lens_k=cu_seqlens,
        is_causal=False,
    )

    assert calls == [2, 3]
    assert attn_weights is None
    assert torch.equal(output, query.transpose(1, 2).contiguous() + 1)


def test_select_varlen_launch_config_uses_stable_defaults_without_experimental_flag(monkeypatch):
    monkeypatch.delenv("TRITON_QWEN3VL_HIGH_PARALLELISM", raising=False)
    query = torch.randn(1, 16, 1152, 128)
    block_m, block_n, num_warps, num_stages = _select_varlen_launch_config(
        query,
        max_length_q=512,
        max_length_k=512,
    )

    assert block_m == 128
    assert block_n == 128
    assert num_warps == 8
    assert num_stages == 2


def test_select_varlen_launch_config_prefers_smaller_tiles_for_packed_short_sequences(monkeypatch):
    monkeypatch.setenv("TRITON_QWEN3VL_HIGH_PARALLELISM", "1")
    query = torch.randn(1, 16, 1152, 128)
    block_m, block_n, num_warps, num_stages = _select_varlen_launch_config(
        query,
        max_length_q=512,
        max_length_k=512,
    )

    assert block_m == 32
    assert block_n == 64
    assert num_warps == 4
    assert num_stages == 3


def test_vision_triton_attention_uses_varlen_kernel_fast_path(monkeypatch):
    sentinel = torch.arange(1 * 2 * 5 * 4, dtype=torch.float32).reshape(1, 2, 5, 4)
    recorded = {}

    monkeypatch.setattr("triton_qwen3vl.backends.can_use_varlen_flash_attention", lambda *args, **kwargs: True)

    def fake_varlen(query, key, value, **kwargs):
        del query, key, value
        recorded.update(kwargs)
        return sentinel

    monkeypatch.setattr("triton_qwen3vl.backends.varlen_flash_attention", fake_varlen)

    module = DummyAttentionModule(num_key_value_groups=1, scaling=0.25, is_causal=False)
    query = torch.randn(1, 2, 5, 4)
    key = torch.randn(1, 2, 5, 4)
    value = torch.randn(1, 2, 5, 4)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)

    output, attn_weights = vision_triton_attention(
        module,
        query,
        key,
        value,
        attention_mask=None,
        dropout=0.0,
        scaling=None,
        cu_seq_lens_q=cu_seqlens,
        cu_seq_lens_k=cu_seqlens,
        max_length_q=3,
        max_length_k=3,
        is_causal=False,
    )

    assert recorded["sm_scale"] == module.scaling
    assert recorded["max_length_q"] == 3
    assert recorded["max_length_k"] == 3
    assert isinstance(recorded["metadata_cache"], dict)
    assert hasattr(module, "_qwen3vl_vision_varlen_metadata_cache")
    assert attn_weights is None
    assert torch.equal(output, sentinel.transpose(1, 2).contiguous())


CUDA_TRITON_AVAILABLE = TRITON_AVAILABLE and torch.cuda.is_available()


@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA Triton environment is required")
def test_gqa_flash_attention_backward_matches_sdpa():
    torch.manual_seed(0)
    q = torch.randn(1, 4, 32, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(1, 2, 32, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(1, 2, 32, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    grad = torch.randn_like(q)
    scale = 64 ** -0.5

    out = gqa_flash_attention(q, k, v, causal=True, sm_scale=scale, num_key_value_groups=2)
    out.backward(grad)

    ref = torch.nn.functional.scaled_dot_product_attention(
        q_ref,
        repeat_kv(k_ref, 2),
        repeat_kv(v_ref, 2),
        is_causal=True,
        scale=scale,
    )
    ref.backward(grad)

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(q.grad, q_ref.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA Triton environment is required")
def test_gqa_decode_attention_backward_matches_sdpa():
    torch.manual_seed(1)
    q = torch.randn(1, 4, 1, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(1, 2, 65, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(1, 2, 65, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    grad = torch.randn_like(q)
    scale = 64 ** -0.5

    out = gqa_decode_attention(q, k, v, sm_scale=scale, num_key_value_groups=2)
    out.backward(grad)

    ref = torch.nn.functional.scaled_dot_product_attention(
        q_ref,
        repeat_kv(k_ref, 2),
        repeat_kv(v_ref, 2),
        is_causal=False,
        scale=scale,
    )
    ref.backward(grad)

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(q.grad, q_ref.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA Triton environment is required")
def test_varlen_flash_attention_backward_matches_chunked_sdpa():
    torch.manual_seed(2)
    q = torch.randn(1, 2, 9, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(1, 2, 9, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(1, 2, 9, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    cu = torch.tensor([0, 2, 5, 9], device="cuda", dtype=torch.int32)
    grad = torch.randn_like(q)
    scale = 64 ** -0.5

    out = varlen_flash_attention(
        q,
        k,
        v,
        cu_seq_lens_q=cu,
        cu_seq_lens_k=cu,
        sm_scale=scale,
        max_length_q=4,
        max_length_k=4,
    )
    out.backward(grad)

    refs = []
    for start, end in zip(cu[:-1].tolist(), cu[1:].tolist()):
        refs.append(
            torch.nn.functional.scaled_dot_product_attention(
                q_ref[:, :, start:end],
                k_ref[:, :, start:end],
                v_ref[:, :, start:end],
                is_causal=False,
                scale=scale,
            )
        )
    ref = torch.cat(refs, dim=2)
    ref.backward(grad)

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(q.grad, q_ref.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=2e-2, rtol=2e-2)
