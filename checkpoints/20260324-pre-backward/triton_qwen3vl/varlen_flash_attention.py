from __future__ import annotations

import torch

from .dense_flash_attention import SUPPORTED_HEAD_DIMS, TRITON_AVAILABLE, tl, triton


def _as_int(value: int | torch.Tensor | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def can_use_varlen_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    cu_seq_lens_q: torch.Tensor | None,
    cu_seq_lens_k: torch.Tensor | None,
    causal: bool,
) -> bool:
    if not TRITON_AVAILABLE:
        return False
    if causal:
        return False
    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        return False
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        return False
    if query.shape[0] != 1 or key.shape[0] != 1 or value.shape[0] != 1:
        return False
    if query.shape[1] != key.shape[1] or key.shape[1] != value.shape[1]:
        return False
    if key.shape[2] != value.shape[2]:
        return False
    if query.shape[-1] != key.shape[-1] or key.shape[-1] != value.shape[-1]:
        return False
    if query.shape[-1] not in SUPPORTED_HEAD_DIMS:
        return False
    if query.dtype not in (torch.float16, torch.bfloat16):
        return False
    if key.dtype != query.dtype or value.dtype != query.dtype:
        return False
    if cu_seq_lens_q is None or cu_seq_lens_k is None:
        return False
    if cu_seq_lens_q.ndim != 1 or cu_seq_lens_k.ndim != 1:
        return False
    if cu_seq_lens_q.numel() != cu_seq_lens_k.numel():
        return False
    if _as_int(cu_seq_lens_q[-1]) != query.shape[2]:
        return False
    if _as_int(cu_seq_lens_k[-1]) != key.shape[2]:
        return False
    return True


def _build_varlen_block_metadata(
    *,
    cu_seq_lens_q: torch.Tensor,
    cu_seq_lens_k: torch.Tensor,
    block_m: int,
    device: torch.device,
    cache: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_lengths = (cu_seq_lens_q[1:] - cu_seq_lens_q[:-1]).detach().cpu().tolist()
    k_lengths = (cu_seq_lens_k[1:] - cu_seq_lens_k[:-1]).detach().cpu().tolist()
    q_offsets = cu_seq_lens_q[:-1].detach().cpu().tolist()
    k_offsets = cu_seq_lens_k[:-1].detach().cpu().tolist()
    cache_key = (tuple(q_lengths), tuple(k_lengths), block_m, device.type, device.index)

    if cache is not None and cache_key in cache:
        return cache[cache_key]

    q_block_starts: list[int] = []
    k_seq_starts: list[int] = []
    q_block_lens: list[int] = []
    k_seq_lens: list[int] = []

    for q_offset, q_length, k_offset, k_length in zip(q_offsets, q_lengths, k_offsets, k_lengths):
        for block_start in range(0, int(q_length), block_m):
            q_block_starts.append(int(q_offset) + block_start)
            k_seq_starts.append(int(k_offset))
            q_block_lens.append(min(block_m, int(q_length) - block_start))
            k_seq_lens.append(int(k_length))

    metadata = (
        torch.tensor(q_block_starts, device=device, dtype=torch.int32),
        torch.tensor(k_seq_starts, device=device, dtype=torch.int32),
        torch.tensor(q_block_lens, device=device, dtype=torch.int32),
        torch.tensor(k_seq_lens, device=device, dtype=torch.int32),
    )

    if cache is not None:
        if len(cache) >= 16:
            oldest_key = next(iter(cache))
            cache.pop(oldest_key, None)
        cache[cache_key] = metadata

    return metadata


if TRITON_AVAILABLE:

    @triton.jit
    def _varlen_flash_attention_fwd(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        q_block_starts_ptr,
        k_seq_starts_ptr,
        q_block_lens_ptr,
        k_seq_lens_ptr,
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
        num_heads,
        max_kv_len,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_block = tl.program_id(0)
        pid_bh = tl.program_id(1)

        off_b = pid_bh // num_heads
        off_h = pid_bh % num_heads

        q_block_start = tl.load(q_block_starts_ptr + pid_block)
        k_seq_start = tl.load(k_seq_starts_ptr + pid_block)
        q_block_len = tl.load(q_block_lens_ptr + pid_block)
        k_seq_len = tl.load(k_seq_lens_ptr + pid_block)

        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, HEAD_DIM)

        q_pos = q_block_start + offs_m
        q_mask = offs_m < q_block_len

        q_ptrs = (
            q_ptr
            + off_b * stride_qz
            + off_h * stride_qh
            + q_pos[:, None] * stride_qm
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

        for start_n in tl.range(0, max_kv_len, BLOCK_N):
            offs_n_curr = start_n + offs_n
            kv_mask = offs_n_curr < k_seq_len
            kv_pos = k_seq_start + offs_n_curr

            k_ptrs = (
                k_ptr
                + off_b * stride_kz
                + off_h * stride_kh
                + kv_pos[:, None] * stride_km
                + offs_d[None, :] * stride_kk
            )
            k = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)

            qk = tl.dot(q, tl.trans(k)) * scale_log2
            attn_mask = q_mask[:, None] & kv_mask[None, :]
            qk = tl.where(attn_mask, qk, -1.0e6)

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            alpha = tl.exp2(m_i - m_ij)
            acc = acc * alpha[:, None]

            v_ptrs = (
                v_ptr
                + off_b * stride_vz
                + off_h * stride_vh
                + kv_pos[:, None] * stride_vm
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
            + q_pos[:, None] * stride_om
            + offs_d[None, :] * stride_ok
        )
        tl.store(o_ptrs, out, mask=q_mask[:, None])


def varlen_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    cu_seq_lens_q: torch.Tensor,
    cu_seq_lens_k: torch.Tensor,
    sm_scale: float,
    max_length_q: int | torch.Tensor | None = None,
    max_length_k: int | torch.Tensor | None = None,
    metadata_cache: dict | None = None,
) -> torch.Tensor:
    if not can_use_varlen_flash_attention(
        query,
        key,
        value,
        cu_seq_lens_q=cu_seq_lens_q,
        cu_seq_lens_k=cu_seq_lens_k,
        causal=False,
    ):
        raise ValueError("Varlen Triton Flash Attention only supports non-causal packed CUDA vision attention in phase 2.")

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    output = torch.empty_like(query)

    max_q = _as_int(max_length_q)
    max_k = _as_int(max_length_k)
    block_m = 64 if (max_q is None or max_q < 256) else 128
    block_n = 64 if (max_k is None or max_k < 512) else 128
    num_warps = 4 if query.shape[-1] <= 64 else 8

    q_block_starts, k_seq_starts, q_block_lens, k_seq_lens = _build_varlen_block_metadata(
        cu_seq_lens_q=cu_seq_lens_q,
        cu_seq_lens_k=cu_seq_lens_k,
        block_m=block_m,
        device=query.device,
        cache=metadata_cache,
    )

    max_kv_len = max_k if max_k is not None else int(k_seq_lens.max().item())
    grid = (q_block_starts.numel(), query.shape[0] * query.shape[1], 1)

    _varlen_flash_attention_fwd[grid](
        query,
        key,
        value,
        output,
        q_block_starts,
        k_seq_starts,
        q_block_lens,
        k_seq_lens,
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
        max_kv_len,
        HEAD_DIM=query.shape[3],
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=2,
    )
    return output
