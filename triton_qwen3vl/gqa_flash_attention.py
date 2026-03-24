from __future__ import annotations

import os

import torch

from .dense_flash_attention import SUPPORTED_HEAD_DIMS, TRITON_AVAILABLE, tl, triton
from .dense_flash_attention_training import dense_flash_attention_training
from .fallback import repeat_kv


def can_use_gqa_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool,
    num_key_value_groups: int,
) -> bool:
    if not TRITON_AVAILABLE:
        return False
    if not causal:
        return False
    if num_key_value_groups < 1:
        return False
    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        return False
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        return False
    if query.shape[0] != key.shape[0] or key.shape[0] != value.shape[0]:
        return False
    if query.shape[2] != key.shape[2] or key.shape[2] != value.shape[2]:
        return False
    if key.shape[1] != value.shape[1]:
        return False
    if query.shape[1] != key.shape[1] * num_key_value_groups:
        return False
    if query.shape[-1] != key.shape[-1] or key.shape[-1] != value.shape[-1]:
        return False
    if query.shape[-1] not in SUPPORTED_HEAD_DIMS:
        return False
    if query.dtype not in (torch.float16, torch.bfloat16):
        return False
    if key.dtype != query.dtype or value.dtype != query.dtype:
        return False
    return True


def _select_gqa_prefill_launch_config(query: torch.Tensor) -> tuple[int, int, int, int]:
    if os.getenv("TRITON_QWEN3VL_HIGH_PARALLELISM", "0") != "1":
        block_m = 64 if query.shape[2] < 256 else 128
        block_n = 64
        num_warps = 4 if query.shape[3] <= 64 else 8
        return block_m, block_n, num_warps, 2

    seq_len = int(query.shape[2])
    head_dim = int(query.shape[3])
    head_parallelism = int(query.shape[0] * query.shape[1])

    if seq_len <= 256:
        block_m = 64
    elif seq_len <= 2048 and head_parallelism <= 32:
        block_m = 64
    else:
        block_m = 128

    block_n = 64
    if head_dim <= 64:
        num_warps = 4
    else:
        num_warps = 4 if block_m == 64 else 8

    num_stages = 3 if seq_len >= 1024 else 2
    return block_m, block_n, num_warps, num_stages


if TRITON_AVAILABLE:

    @triton.jit
    def _gqa_flash_attention_fwd(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_km,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vm,
        stride_vk,
        stride_oz,
        stride_oh,
        stride_om,
        stride_ok,
        sm_scale,
        num_query_heads,
        seq_len,
        num_key_value_groups,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)

        off_b = pid_bh // num_query_heads
        off_hq = pid_bh % num_query_heads
        off_hkv = off_hq // num_key_value_groups

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, HEAD_DIM)

        q_mask = offs_m < seq_len
        q_ptrs = (
            q_ptr
            + off_b * stride_qz
            + off_hq * stride_qh
            + offs_m[:, None] * stride_qm
            + offs_d[None, :] * stride_qk
        )
        q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)

        m_i = tl.where(q_mask, tl.full([BLOCK_M], -1.0e6, dtype=tl.float32), tl.zeros([BLOCK_M], dtype=tl.float32))
        l_i = tl.where(
            q_mask,
            tl.zeros([BLOCK_M], dtype=tl.float32),
            tl.full([BLOCK_M], 1.0, dtype=tl.float32),
        )
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        scale_log2 = sm_scale * 1.4426950408889634

        upper_bound = seq_len
        if IS_CAUSAL:
            upper_bound = tl.minimum(seq_len, (pid_m + 1) * BLOCK_M)

        for start_n in tl.range(0, upper_bound, BLOCK_N):
            offs_n_curr = start_n + offs_n
            kv_mask = offs_n_curr < seq_len

            k_ptrs = (
                k_ptr
                + off_b * stride_kz
                + off_hkv * stride_kh
                + offs_n_curr[:, None] * stride_km
                + offs_d[None, :] * stride_kk
            )
            k = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)

            qk = tl.dot(q, tl.trans(k)) * scale_log2
            attn_mask = q_mask[:, None] & kv_mask[None, :]
            if IS_CAUSAL:
                attn_mask = attn_mask & (offs_m[:, None] >= offs_n_curr[None, :])
            qk = tl.where(attn_mask, qk, -1.0e6)

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            alpha = tl.exp2(m_i - m_ij)
            acc = acc * alpha[:, None]

            v_ptrs = (
                v_ptr
                + off_b * stride_vz
                + off_hkv * stride_vh
                + offs_n_curr[:, None] * stride_vm
                + offs_d[None, :] * stride_vk
            )
            v = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)

            acc = tl.dot(p.to(v.dtype), v, acc)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_ij

        denom = tl.where(q_mask, l_i, tl.full([BLOCK_M], 1.0, dtype=tl.float32))
        out = acc / denom[:, None]
        o_ptrs = (
            o_ptr
            + off_b * stride_oz
            + off_hq * stride_oh
            + offs_m[:, None] * stride_om
            + offs_d[None, :] * stride_ok
        )
        tl.store(o_ptrs, out, mask=q_mask[:, None])


def _gqa_flash_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool,
    sm_scale: float,
    num_key_value_groups: int,
) -> torch.Tensor:
    if not can_use_gqa_flash_attention(
        query,
        key,
        value,
        causal=causal,
        num_key_value_groups=num_key_value_groups,
    ):
        raise ValueError("GQA Triton Flash Attention only supports square CUDA fp16/bf16 causal prefill attention.")

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    output = torch.empty_like(query)
    block_m, block_n, num_warps, num_stages = _select_gqa_prefill_launch_config(query)

    grid = lambda meta: (triton.cdiv(query.shape[2], meta["BLOCK_M"]), query.shape[0] * query.shape[1], 1)

    _gqa_flash_attention_fwd[grid](
        query,
        key,
        value,
        output,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        sm_scale,
        query.shape[1],
        query.shape[2],
        num_key_value_groups,
        HEAD_DIM=query.shape[3],
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        IS_CAUSAL=causal,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


def gqa_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool,
    sm_scale: float,
    num_key_value_groups: int,
) -> torch.Tensor:
    if not causal:
        raise ValueError("gqa_flash_attention only supports causal prefill attention.")

    if torch.is_grad_enabled() and (query.requires_grad or key.requires_grad or value.requires_grad):
        return dense_flash_attention_training(
            query,
            repeat_kv(key, num_key_value_groups),
            repeat_kv(value, num_key_value_groups),
            sm_scale=sm_scale,
            causal=True,
        )

    return _gqa_flash_attention_forward(
        query,
        key,
        value,
        causal=causal,
        sm_scale=sm_scale,
        num_key_value_groups=num_key_value_groups,
    )
