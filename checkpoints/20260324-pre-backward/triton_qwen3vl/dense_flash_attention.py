from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None
    TRITON_AVAILABLE = False
else:
    TRITON_AVAILABLE = True


SUPPORTED_HEAD_DIMS = {16, 32, 64, 128}


def can_use_dense_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool,
) -> bool:
    if not TRITON_AVAILABLE:
        return False
    if not causal:
        return False
    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        return False
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        return False
    if query.shape != key.shape or key.shape != value.shape:
        return False
    if query.shape[-1] not in SUPPORTED_HEAD_DIMS:
        return False
    if query.dtype not in (torch.float16, torch.bfloat16):
        return False
    if key.dtype != query.dtype or value.dtype != query.dtype:
        return False
    return True


if TRITON_AVAILABLE:

    @triton.jit
    def _dense_flash_attention_fwd(
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
        batch_size,
        num_heads,
        seq_len,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)

        off_b = pid_bh // num_heads
        off_h = pid_bh % num_heads

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, HEAD_DIM)

        q_mask = offs_m < seq_len
        q_ptrs = (
            q_ptr
            + off_b * stride_qz
            + off_h * stride_qh
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
                + off_h * stride_kh
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
                + off_h * stride_vh
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
            + off_h * stride_oh
            + offs_m[:, None] * stride_om
            + offs_d[None, :] * stride_ok
        )
        tl.store(o_ptrs, out, mask=q_mask[:, None])


def dense_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool,
    sm_scale: float,
) -> torch.Tensor:
    if not can_use_dense_flash_attention(query, key, value, causal=causal):
        raise ValueError("Dense Triton Flash Attention only supports square CUDA fp16/bf16 attention in phase 1.")

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    output = torch.empty_like(query)
    block_m = 64 if query.shape[2] < 256 else 128
    block_n = 64
    num_warps = 4 if query.shape[3] <= 64 else 8

    grid = lambda meta: (triton.cdiv(query.shape[2], meta["BLOCK_M"]), query.shape[0] * query.shape[1], 1)

    _dense_flash_attention_fwd[grid](
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
        query.shape[0],
        query.shape[1],
        query.shape[2],
        HEAD_DIM=query.shape[3],
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        IS_CAUSAL=causal,
        num_warps=num_warps,
        num_stages=2,
    )
    return output
