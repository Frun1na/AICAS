from __future__ import annotations

import warnings

import torch

from .dense_flash_attention import TRITON_AVAILABLE
from .fallback import sdpa_fallback
from .gqa_decode_attention import can_use_gqa_decode_attention, gqa_decode_attention
from .gqa_flash_attention import can_use_gqa_flash_attention, gqa_flash_attention
from .varlen_flash_attention import can_use_varlen_flash_attention, varlen_flash_attention


TEXT_BACKEND_KEY = "qwen3vl_triton_flash_text"
VISION_BACKEND_KEY = "qwen3vl_triton_flash_vision"


def _warn_once(module: torch.nn.Module, attr_name: str, message: str) -> None:
    if getattr(module, attr_name, False):
        return
    warnings.warn(message)
    setattr(module, attr_name, True)


def _resolve_is_causal(module: torch.nn.Module, is_causal: bool | None) -> bool:
    if is_causal is not None:
        return is_causal
    return bool(getattr(module, "is_causal", True))


def _resolve_scale(module: torch.nn.Module, scaling: float | None) -> float:
    if scaling is not None:
        return scaling
    return float(getattr(module, "scaling"))


def _split_lengths(cu_seq_lens: torch.Tensor) -> list[int]:
    return (cu_seq_lens[1:] - cu_seq_lens[:-1]).detach().cpu().tolist()


def _vision_varlen_fallback(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    dropout: float,
    scaling: float | None,
    cu_seq_lens_q: torch.Tensor | None,
    cu_seq_lens_k: torch.Tensor | None,
    is_causal: bool,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if cu_seq_lens_q is None or cu_seq_lens_k is None:
        return sdpa_fallback(
            module,
            query,
            key,
            value,
            attention_mask=None,
            dropout=dropout,
            scaling=scaling,
            is_causal=is_causal,
            **kwargs,
        )

    q_lengths = _split_lengths(cu_seq_lens_q)
    k_lengths = _split_lengths(cu_seq_lens_k)
    q_chunks = torch.split(query, q_lengths, dim=2)
    k_chunks = torch.split(key, k_lengths, dim=2)
    v_chunks = torch.split(value, k_lengths, dim=2)

    outputs = []
    for q_chunk, k_chunk, v_chunk in zip(q_chunks, k_chunks, v_chunks):
        out, _ = sdpa_fallback(
            module,
            q_chunk,
            k_chunk,
            v_chunk,
            attention_mask=None,
            dropout=dropout,
            scaling=scaling,
            is_causal=is_causal,
            **kwargs,
        )
        outputs.append(out)
    return torch.cat(outputs, dim=1), None


def vision_triton_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    dropout: float = 0.0,
    scaling: float | None = None,
    cu_seq_lens_q: torch.Tensor | None = None,
    cu_seq_lens_k: torch.Tensor | None = None,
    max_length_q: int | torch.Tensor | None = None,
    max_length_k: int | torch.Tensor | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    del attention_mask
    is_causal = _resolve_is_causal(module, is_causal)
    sm_scale = _resolve_scale(module, scaling)

    if kwargs.get("output_attentions", False):
        return _vision_varlen_fallback(
            module,
            query,
            key,
            value,
            dropout=dropout,
            scaling=scaling,
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_k=cu_seq_lens_k,
            is_causal=is_causal,
            **kwargs,
        )

    can_use_triton = (
        dropout == 0.0
        and can_use_varlen_flash_attention(
            query,
            key,
            value,
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_k=cu_seq_lens_k,
            causal=is_causal,
        )
    )
    if can_use_triton:
        try:
            metadata_cache = getattr(module, "_qwen3vl_vision_varlen_metadata_cache", None)
            if metadata_cache is None:
                metadata_cache = {}
                setattr(module, "_qwen3vl_vision_varlen_metadata_cache", metadata_cache)
            attn_output = varlen_flash_attention(
                query,
                key,
                value,
                cu_seq_lens_q=cu_seq_lens_q,
                cu_seq_lens_k=cu_seq_lens_k,
                sm_scale=sm_scale,
                max_length_q=max_length_q,
                max_length_k=max_length_k,
                metadata_cache=metadata_cache,
            )
            return attn_output.transpose(1, 2).contiguous(), None
        except Exception as exc:
            _warn_once(
                module,
                "_qwen3vl_vision_triton_failed_warned",
                f"[Qwen3-VL Triton] Vision varlen Triton kernel failed, falling back to SDPA: {exc}",
            )

    if TRITON_AVAILABLE and cu_seq_lens_q is not None and cu_seq_lens_k is not None:
        _warn_once(
            module,
            "_qwen3vl_vision_triton_fallback_warned",
            "[Qwen3-VL Triton] Vision varlen attention fell back to SDPA because the current inputs are not on the Triton fast path.",
        )
    return _vision_varlen_fallback(
        module,
        query,
        key,
        value,
        dropout=dropout,
        scaling=sm_scale,
        cu_seq_lens_q=cu_seq_lens_q,
        cu_seq_lens_k=cu_seq_lens_k,
        is_causal=is_causal,
        **kwargs,
    )


def text_triton_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    is_causal = _resolve_is_causal(module, is_causal)
    sm_scale = _resolve_scale(module, scaling)
    num_key_value_groups = int(getattr(module, "num_key_value_groups", 1))

    if kwargs.get("output_attentions", False):
        return sdpa_fallback(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=sm_scale,
            is_causal=is_causal,
            **kwargs,
        )

    can_use_decode = (
        attention_mask is None
        and query.shape[2] == 1
        and dropout == 0.0
        and can_use_gqa_decode_attention(
            query,
            key,
            value,
            num_key_value_groups=num_key_value_groups,
        )
    )
    if can_use_decode:
        try:
            attn_output = gqa_decode_attention(
                query,
                key,
                value,
                sm_scale=sm_scale,
                num_key_value_groups=num_key_value_groups,
            )
            return attn_output.transpose(1, 2).contiguous(), None
        except Exception as exc:
            _warn_once(
                module,
                "_qwen3vl_text_decode_triton_failed_warned",
                f"[Qwen3-VL Triton] Text decode Triton kernel failed, falling back to SDPA: {exc}",
            )

    can_use_prefill = (
        attention_mask is None
        and query.shape[2] > 1
        and query.shape[2] == key.shape[2]
        and dropout == 0.0
        and can_use_gqa_flash_attention(
            query,
            key,
            value,
            causal=is_causal,
            num_key_value_groups=num_key_value_groups,
        )
    )

    if not can_use_prefill:
        return sdpa_fallback(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=sm_scale,
            is_causal=is_causal,
            **kwargs,
        )

    try:
        attn_output = gqa_flash_attention(
            query,
            key,
            value,
            causal=is_causal,
            sm_scale=sm_scale,
            num_key_value_groups=num_key_value_groups,
        )
    except Exception as exc:
        _warn_once(
            module,
            "_qwen3vl_text_triton_failed_warned",
            f"[Qwen3-VL Triton] Text GQA Triton kernel failed, falling back to SDPA: {exc}",
        )
        return sdpa_fallback(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=sm_scale,
            is_causal=is_causal,
            **kwargs,
        )

    return attn_output.transpose(1, 2).contiguous(), None
