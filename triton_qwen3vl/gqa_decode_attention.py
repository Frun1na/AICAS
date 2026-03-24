from __future__ import annotations

import torch

from .dense_flash_attention import SUPPORTED_HEAD_DIMS, TRITON_AVAILABLE, tl, triton


def can_use_gqa_decode_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    num_key_value_groups: int,
) -> bool:
    if not TRITON_AVAILABLE:
        return False
    if num_key_value_groups < 1:
        return False
    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        return False
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        return False
    if query.shape[0] != key.shape[0] or key.shape[0] != value.shape[0]:
        return False
    if query.shape[2] != 1:
        return False
    if key.shape[2] != value.shape[2]:
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


if TRITON_AVAILABLE:

    @triton.jit
    def _gqa_decode_attention_fwd(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        lse_ptr,
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
        stride_lsez,
        stride_lseh,
        sm_scale,
        num_query_heads,
        kv_len,
        num_key_value_groups,
        HEAD_DIM: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)

        off_b = pid_bh // num_query_heads
        off_hq = pid_bh % num_query_heads
        off_hkv = off_hq // num_key_value_groups

        offs_d = tl.arange(0, HEAD_DIM)
        offs_n = tl.arange(0, BLOCK_N)

        q_ptrs = (
            q_ptr
            + off_b * stride_qz
            + off_hq * stride_qh
            + offs_d * stride_qk
        )
        q = tl.load(q_ptrs)

        m_i = tl.full((), -1.0e6, dtype=tl.float32)
        l_i = tl.full((), 0.0, dtype=tl.float32)
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
        scale_log2 = sm_scale * 1.4426950408889634

        for start_n in tl.range(0, kv_len, BLOCK_N):
            offs_n_curr = start_n + offs_n
            kv_mask = offs_n_curr < kv_len

            k_ptrs = (
                k_ptr
                + off_b * stride_kz
                + off_hkv * stride_kh
                + offs_n_curr[:, None] * stride_km
                + offs_d[None, :] * stride_kk
            )
            k = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)
            qk = tl.sum(k * q[None, :], axis=1) * scale_log2
            qk = tl.where(kv_mask, qk, -1.0e6)

            m_ij = tl.maximum(m_i, tl.max(qk, axis=0))
            p = tl.exp2(qk - m_ij)
            alpha = tl.exp2(m_i - m_ij)
            acc = acc * alpha

            v_ptrs = (
                v_ptr
                + off_b * stride_vz
                + off_hkv * stride_vh
                + offs_n_curr[:, None] * stride_vm
                + offs_d[None, :] * stride_vk
            )
            v = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)
            acc = acc + tl.sum(v * p[:, None].to(v.dtype), axis=0)

            l_i = l_i * alpha + tl.sum(p, axis=0)
            m_i = m_ij

        out = acc / l_i
        o_ptrs = (
            o_ptr
            + off_b * stride_oz
            + off_hq * stride_oh
            + offs_d * stride_ok
        )
        tl.store(o_ptrs, out)
        tl.store(lse_ptr + off_b * stride_lsez + off_hq * stride_lseh, m_i + tl.math.log2(l_i))

    @triton.jit
    def _gqa_decode_attention_bwd(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        do_ptr,
        lse_ptr,
        dq_ptr,
        dk_ptr,
        dv_ptr,
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
        stride_doz,
        stride_doh,
        stride_dom,
        stride_dok,
        stride_lsez,
        stride_lseh,
        stride_dqz,
        stride_dqh,
        stride_dqm,
        stride_dqk,
        stride_dkz,
        stride_dkh,
        stride_dkm,
        stride_dkk,
        stride_dvz,
        stride_dvh,
        stride_dvm,
        stride_dvk,
        sm_scale,
        num_query_heads,
        kv_len,
        num_key_value_groups,
        HEAD_DIM: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)

        off_b = pid_bh // num_query_heads
        off_hq = pid_bh % num_query_heads
        off_hkv = off_hq // num_key_value_groups

        offs_d = tl.arange(0, HEAD_DIM)
        offs_n = tl.arange(0, BLOCK_N)

        q = tl.load(q_ptr + off_b * stride_qz + off_hq * stride_qh + offs_d * stride_qk).to(tl.float32)
        do = tl.load(do_ptr + off_b * stride_doz + off_hq * stride_doh + offs_d * stride_dok).to(tl.float32)
        o = tl.load(o_ptr + off_b * stride_oz + off_hq * stride_oh + offs_d * stride_ok).to(tl.float32)
        lse = tl.load(lse_ptr + off_b * stride_lsez + off_hq * stride_lseh)
        delta = tl.sum(do * o, axis=0)

        dq = tl.zeros([HEAD_DIM], dtype=tl.float32)
        scale_log2 = sm_scale * 1.4426950408889634

        for start_n in tl.range(0, kv_len, BLOCK_N):
            offs_n_curr = start_n + offs_n
            kv_mask = offs_n_curr < kv_len

            k_ptrs = (
                k_ptr
                + off_b * stride_kz
                + off_hkv * stride_kh
                + offs_n_curr[:, None] * stride_km
                + offs_d[None, :] * stride_kk
            )
            v_ptrs = (
                v_ptr
                + off_b * stride_vz
                + off_hkv * stride_vh
                + offs_n_curr[:, None] * stride_vm
                + offs_d[None, :] * stride_vk
            )

            k = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)
            v = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)

            logits = tl.sum(k * q[None, :], axis=1) * scale_log2
            p = tl.where(kv_mask, tl.exp2(logits - lse), 0.0)
            dp = tl.sum(v * do[None, :], axis=1)
            ds = p * (dp - delta) * sm_scale

            dq += tl.sum(ds[:, None] * k, axis=0)

            dk_ptrs = (
                dk_ptr
                + off_b * stride_dkz
                + off_hkv * stride_dkh
                + offs_n_curr[:, None] * stride_dkm
                + offs_d[None, :] * stride_dkk
            )
            dv_ptrs = (
                dv_ptr
                + off_b * stride_dvz
                + off_hkv * stride_dvh
                + offs_n_curr[:, None] * stride_dvm
                + offs_d[None, :] * stride_dvk
            )
            tl.atomic_add(dk_ptrs, ds[:, None] * q[None, :], mask=kv_mask[:, None])
            tl.atomic_add(dv_ptrs, p[:, None] * do[None, :], mask=kv_mask[:, None])

        dq_ptrs = (
            dq_ptr
            + off_b * stride_dqz
            + off_hq * stride_dqh
            + offs_d * stride_dqk
        )
        tl.store(dq_ptrs, dq)


def _gqa_decode_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    sm_scale: float,
    num_key_value_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not can_use_gqa_decode_attention(
        query,
        key,
        value,
        num_key_value_groups=num_key_value_groups,
    ):
        raise ValueError("GQA Triton decode attention only supports CUDA fp16/bf16 q_len==1 inference.")

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    output = torch.empty_like(query)
    lse = torch.empty((query.shape[0], query.shape[1]), device=query.device, dtype=torch.float32)
    block_n = 64 if key.shape[2] < 512 else 128
    num_warps = 4 if query.shape[3] <= 64 else 8

    grid = (query.shape[0] * query.shape[1], 1, 1)

    _gqa_decode_attention_fwd[grid](
        query,
        key,
        value,
        output,
        lse,
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
        lse.stride(0),
        lse.stride(1),
        sm_scale,
        query.shape[1],
        key.shape[2],
        num_key_value_groups,
        HEAD_DIM=query.shape[3],
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=2,
    )
    return output, lse


class _GQADecodeAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, sm_scale, num_key_value_groups):
        ctx.sm_scale = float(sm_scale)
        ctx.num_key_value_groups = int(num_key_value_groups)
        output, lse = _gqa_decode_attention_forward(
            query,
            key,
            value,
            sm_scale=ctx.sm_scale,
            num_key_value_groups=ctx.num_key_value_groups,
        )
        ctx.save_for_backward(query, key, value, output, lse)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        query, key, value, output, lse = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        dq = torch.empty_like(query, dtype=torch.float32)
        dk = torch.zeros_like(key, dtype=torch.float32)
        dv = torch.zeros_like(value, dtype=torch.float32)
        block_n = 64 if key.shape[2] < 512 else 128
        num_warps = 4 if query.shape[3] <= 64 else 8

        grid = (query.shape[0] * query.shape[1], 1, 1)
        _gqa_decode_attention_bwd[grid](
            query,
            key,
            value,
            output,
            grad_output,
            lse,
            dq,
            dk,
            dv,
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
            grad_output.stride(0),
            grad_output.stride(1),
            grad_output.stride(2),
            grad_output.stride(3),
            lse.stride(0),
            lse.stride(1),
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            dq.stride(3),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            dk.stride(3),
            dv.stride(0),
            dv.stride(1),
            dv.stride(2),
            dv.stride(3),
            ctx.sm_scale,
            query.shape[1],
            key.shape[2],
            ctx.num_key_value_groups,
            HEAD_DIM=query.shape[3],
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=2,
        )

        return dq.to(query.dtype), dk.to(key.dtype), dv.to(value.dtype), None, None


def gqa_decode_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    sm_scale: float,
    num_key_value_groups: int,
) -> torch.Tensor:
    if torch.is_grad_enabled() and (query.requires_grad or key.requires_grad or value.requires_grad):
        return _GQADecodeAttentionFunction.apply(query, key, value, sm_scale, num_key_value_groups)

    return _gqa_decode_attention_forward(
        query,
        key,
        value,
        sm_scale=sm_scale,
        num_key_value_groups=num_key_value_groups,
    )[0]
