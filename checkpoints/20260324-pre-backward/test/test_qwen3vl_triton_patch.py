from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triton_qwen3vl import patch
from triton_qwen3vl.backends import text_triton_attention, vision_triton_attention


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
