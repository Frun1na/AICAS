from __future__ import annotations

import torch

from .dense_flash_attention import SUPPORTED_HEAD_DIMS, TRITON_AVAILABLE, tl, triton


PRE_BLOCK = 128
BLOCK_M1 = 32
BLOCK_N1 = 128
BLOCK_M2 = 128
BLOCK_N2 = 32
BLK_SLICE_FACTOR = 2
NUM_WARPS = 4
NUM_STAGES = 5


def can_use_dense_flash_attention_training(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool,
) -> bool:
    if not TRITON_AVAILABLE:
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


def _pad_to_multiple(tensor: torch.Tensor, *, multiple: int) -> torch.Tensor:
    seq_len = tensor.shape[2]
    padded_len = ((seq_len + multiple - 1) // multiple) * multiple
    if padded_len == seq_len:
        return tensor
    pad = torch.zeros(
        tensor.shape[0],
        tensor.shape[1],
        padded_len - seq_len,
        tensor.shape[3],
        device=tensor.device,
        dtype=tensor.dtype,
    )
    return torch.cat([tensor, pad], dim=2)


if TRITON_AVAILABLE:

    @triton.jit
    def _dense_train_attention_fwd(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        m_ptr,
        stride_mz,
        stride_mh,
        stride_mm,
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
        padded_seq_len,
        valid_seq_len,
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

        q_mask = offs_m < valid_seq_len
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

        upper_bound = valid_seq_len
        if IS_CAUSAL:
            upper_bound = tl.minimum(valid_seq_len, (pid_m + 1) * BLOCK_M)

        for start_n in tl.range(0, upper_bound, BLOCK_N):
            offs_n_curr = start_n + offs_n
            kv_mask = offs_n_curr < valid_seq_len

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

        m_i += tl.math.log2(l_i)
        out = acc / l_i[:, None]
        o_ptrs = (
            o_ptr
            + off_b * stride_oz
            + off_h * stride_oh
            + offs_m[:, None] * stride_om
            + offs_d[None, :] * stride_ok
        )
        tl.store(o_ptrs, out, mask=q_mask[:, None])

        m_store_ptrs = m_ptr + off_b * stride_mz + off_h * stride_mh + offs_m * stride_mm
        tl.store(m_store_ptrs, m_i, mask=q_mask)

    @triton.jit
    def _attn_bwd_preprocess(
        O,
        DO,
        Delta,
        stride_oz,
        stride_oh,
        stride_om,
        stride_ok,
        stride_doz,
        stride_doh,
        stride_dom,
        stride_dok,
        stride_deltaz,
        stride_deltah,
        stride_deltam,
        H,
        padded_seq_len,
        valid_seq_len,
        BLOCK_M: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        off_hz = tl.program_id(1)
        off_b = off_hz // H
        off_h = off_hz % H
        off_n = tl.arange(0, HEAD_DIM)
        q_mask = off_m < valid_seq_len
        o = tl.load(
            O + off_b * stride_oz + off_h * stride_oh + off_m[:, None] * stride_om + off_n[None, :] * stride_ok,
            mask=q_mask[:, None],
            other=0.0,
        )
        do = tl.load(
            DO + off_b * stride_doz + off_h * stride_doh + off_m[:, None] * stride_dom + off_n[None, :] * stride_dok,
            mask=q_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        delta = tl.sum(o * do, axis=1)
        tl.store(
            Delta + off_b * stride_deltaz + off_h * stride_deltah + off_m * stride_deltam,
            delta,
            mask=q_mask,
        )

    @triton.jit
    def _attn_bwd_dkdv(
        dk,
        dv,
        Q,
        k,
        v,
        sm_scale,
        DO,
        M,
        D,
        stride_tok,
        stride_d,
        H,
        N_CTX,
        VALID_SEQ_LEN,
        BLOCK_M1: tl.constexpr,
        BLOCK_N1: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        start_n,
        start_m,
        num_steps,
        MASK: tl.constexpr,
    ):
        offs_m = start_m + tl.arange(0, BLOCK_M1)
        offs_n = start_n + tl.arange(0, BLOCK_N1)
        offs_k = tl.arange(0, HEAD_DIM)
        kv_mask = offs_n < VALID_SEQ_LEN
        qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
        do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
        tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
        curr_m = start_m
        step_m = BLOCK_M1
        for _ in range(num_steps):
            q_mask = offs_m < VALID_SEQ_LEN
            qT = tl.load(qT_ptrs, mask=q_mask[None, :], other=0.0)
            offs_m = curr_m + tl.arange(0, BLOCK_M1)
            q_mask = offs_m < VALID_SEQ_LEN
            m = tl.load(M + offs_m, mask=q_mask, other=0.0)
            qkT = tl.dot(k, qT)
            pT = tl.math.exp2(qkT - m[None, :])
            mask = kv_mask[:, None] & q_mask[None, :]
            if MASK:
                mask = mask & (offs_m[None, :] >= offs_n[:, None])
            pT = tl.where(mask, pT, 0.0)
            do = tl.load(do_ptrs, mask=q_mask[:, None], other=0.0)
            dv += tl.dot(pT.to(tl.float16), do)
            Di = tl.load(D + offs_m, mask=q_mask, other=0.0)
            dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
            dsT = pT * (dpT - Di[None, :])
            dk += tl.dot(dsT.to(tl.float16), tl.trans(qT))
            curr_m += step_m
            qT_ptrs += step_m * stride_tok
            do_ptrs += step_m * stride_tok
        return dk, dv

    @triton.jit
    def _attn_bwd_dq(
        dq,
        q,
        K,
        V,
        do,
        m,
        D,
        stride_tok,
        stride_d,
        H,
        N_CTX,
        VALID_SEQ_LEN,
        BLOCK_M2: tl.constexpr,
        BLOCK_N2: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        start_m,
        start_n,
        num_steps,
        MASK: tl.constexpr,
    ):
        offs_m = start_m + tl.arange(0, BLOCK_M2)
        offs_n = start_n + tl.arange(0, BLOCK_N2)
        offs_k = tl.arange(0, HEAD_DIM)
        kv_mask = offs_n < VALID_SEQ_LEN
        kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
        vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
        q_mask = offs_m < VALID_SEQ_LEN
        Di = tl.load(D + offs_m, mask=q_mask, other=0.0)
        tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
        curr_n = start_n
        step_n = BLOCK_N2
        for _ in range(num_steps):
            kT = tl.load(kT_ptrs, mask=kv_mask[None, :], other=0.0)
            vT = tl.load(vT_ptrs, mask=kv_mask[None, :], other=0.0)
            qk = tl.dot(q, kT)
            p = tl.math.exp2(qk - m)
            mask = q_mask[:, None] & kv_mask[None, :]
            if MASK:
                offs_n = curr_n + tl.arange(0, BLOCK_N2)
                mask = mask & (offs_m[:, None] >= offs_n[None, :])
            p = tl.where(mask, p, 0.0)
            dp = tl.dot(do, vT).to(tl.float32)
            ds = p * (dp - Di[:, None])
            dq += tl.dot(ds.to(tl.float16), tl.trans(kT))
            curr_n += step_n
            kv_mask = (curr_n + tl.arange(0, BLOCK_N2)) < VALID_SEQ_LEN
            kT_ptrs += step_n * stride_tok
            vT_ptrs += step_n * stride_tok
        return dq

    @triton.jit
    def _attn_bwd(
        Q,
        K,
        V,
        sm_scale,
        DO,
        DQ,
        DK,
        DV,
        M,
        D,
        stride_z,
        stride_h,
        stride_tok,
        stride_d,
        stride_mz,
        stride_mh,
        stride_mm,
        H,
        N_CTX,
        VALID_SEQ_LEN,
        BLOCK_M1: tl.constexpr,
        BLOCK_N1: tl.constexpr,
        BLOCK_M2: tl.constexpr,
        BLOCK_N2: tl.constexpr,
        BLK_SLICE_FACTOR: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        CAUSAL: tl.constexpr,
    ):
        LN2: tl.constexpr = 0.6931471824645996
        bhid = tl.program_id(2)
        off_chz = (bhid * N_CTX).to(tl.int64)
        adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
        pid = tl.program_id(0)

        Q += adj
        K += adj
        V += adj
        DO += adj
        DQ += adj
        DK += adj
        DV += adj
        off_b = bhid // H
        off_h = bhid % H
        M += off_b * stride_mz + off_h * stride_mh
        D += off_b * stride_mz + off_h * stride_mh

        offs_k = tl.arange(0, HEAD_DIM)
        start_n = pid * BLOCK_N1
        start_m = 0

        MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
        offs_n = start_n + tl.arange(0, BLOCK_N1)
        kv_mask = offs_n < VALID_SEQ_LEN

        dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
        dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

        k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d, mask=kv_mask[:, None], other=0.0)
        v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d, mask=kv_mask[:, None], other=0.0)

        if CAUSAL:
            start_m = start_n
            num_steps = BLOCK_N1 // MASK_BLOCK_M1
            dk, dv = _attn_bwd_dkdv(
                dk,
                dv,
                Q,
                k,
                v,
                sm_scale,
                DO,
                M,
                D,
                stride_tok,
                stride_d,
                H,
                N_CTX,
                VALID_SEQ_LEN,
                MASK_BLOCK_M1,
                BLOCK_N1,
                HEAD_DIM,
                start_n,
                start_m,
                num_steps,
                MASK=True,
            )
            start_m += num_steps * MASK_BLOCK_M1

        num_steps = (N_CTX - start_m) // BLOCK_M1
        dk, dv = _attn_bwd_dkdv(
            dk,
            dv,
            Q,
            k,
            v,
            sm_scale,
            DO,
            M,
            D,
            stride_tok,
            stride_d,
            H,
            N_CTX,
            VALID_SEQ_LEN,
            BLOCK_M1,
            BLOCK_N1,
            HEAD_DIM,
            start_n,
            start_m,
            num_steps,
            MASK=False,
        )

        dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
        tl.store(dv_ptrs, dv, mask=kv_mask[:, None])

        dk *= sm_scale
        dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
        tl.store(dk_ptrs, dk, mask=kv_mask[:, None])

        start_m = pid * BLOCK_M2
        start_n = 0
        num_steps = N_CTX // BLOCK_N2
        MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
        offs_m = start_m + tl.arange(0, BLOCK_M2)
        q_mask = offs_m < VALID_SEQ_LEN

        q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d, mask=q_mask[:, None], other=0.0)
        dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
        do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d, mask=q_mask[:, None], other=0.0)
        m = tl.load(M + offs_m, mask=q_mask, other=0.0)[:, None]

        if CAUSAL:
            end_n = start_m + BLOCK_M2
            num_steps = BLOCK_M2 // MASK_BLOCK_N2
            dq = _attn_bwd_dq(
                dq,
                q,
                K,
                V,
                do,
                m,
                D,
                stride_tok,
                stride_d,
                H,
                N_CTX,
                VALID_SEQ_LEN,
                BLOCK_M2,
                MASK_BLOCK_N2,
                HEAD_DIM,
                start_m,
                end_n - num_steps * MASK_BLOCK_N2,
                num_steps,
                MASK=True,
            )
            end_n -= num_steps * MASK_BLOCK_N2
            num_steps = end_n // BLOCK_N2
            start_n = end_n - num_steps * BLOCK_N2

        dq = _attn_bwd_dq(
            dq,
            q,
            K,
            V,
            do,
            m,
            D,
            stride_tok,
            stride_d,
            H,
            N_CTX,
            VALID_SEQ_LEN,
            BLOCK_M2,
            BLOCK_N2,
            HEAD_DIM,
            start_m,
            start_n,
            num_steps,
            MASK=False,
        )
        dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
        tl.store(dq_ptrs, dq * LN2, mask=q_mask[:, None])


class _DenseFlashAttentionTrainingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, sm_scale, causal):
        causal = bool(causal)
        if not can_use_dense_flash_attention_training(query, key, value, causal=causal):
            raise ValueError("dense_flash_attention_training requires square CUDA fp16/bf16 attention.")

        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        original_seq_len = query.shape[2]
        query = _pad_to_multiple(query, multiple=PRE_BLOCK)
        key = _pad_to_multiple(key, multiple=PRE_BLOCK)
        value = _pad_to_multiple(value, multiple=PRE_BLOCK)

        output = torch.empty_like(query)
        m = torch.zeros((query.shape[0], query.shape[1], query.shape[2]), device=query.device, dtype=torch.float32)

        grid = (triton.cdiv(query.shape[2], PRE_BLOCK), query.shape[0] * query.shape[1], 1)
        _dense_train_attention_fwd[grid](
            query,
            key,
            value,
            output,
            m,
            m.stride(0),
            m.stride(1),
            m.stride(2),
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
            original_seq_len,
            HEAD_DIM=query.shape[3],
            BLOCK_M=PRE_BLOCK,
            BLOCK_N=64,
            IS_CAUSAL=causal,
            num_warps=NUM_WARPS,
            num_stages=2,
        )

        ctx.sm_scale = float(sm_scale)
        ctx.causal = causal
        ctx.original_seq_len = original_seq_len
        ctx.save_for_backward(query, key, value, output, m)
        return output[:, :, :original_seq_len, :]

    @staticmethod
    def backward(ctx, grad_output):
        query, key, value, output, m = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        if grad_output.shape[2] != query.shape[2]:
            grad_output = _pad_to_multiple(grad_output, multiple=PRE_BLOCK)

        dq = torch.empty_like(query)
        dk = torch.empty_like(key)
        dv = torch.empty_like(value)
        delta = torch.empty_like(m)

        pre_grid = (query.shape[2] // PRE_BLOCK, query.shape[0] * query.shape[1])
        _attn_bwd_preprocess[pre_grid](
            output,
            grad_output,
            delta,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output.stride(3),
            grad_output.stride(0),
            grad_output.stride(1),
            grad_output.stride(2),
            grad_output.stride(3),
            delta.stride(0),
            delta.stride(1),
            delta.stride(2),
            query.shape[1],
            query.shape[2],
            ctx.original_seq_len,
            BLOCK_M=PRE_BLOCK,
            HEAD_DIM=query.shape[3],
        )

        grid = (query.shape[2] // BLOCK_N1, 1, query.shape[0] * query.shape[1])
        arg_k = key * (ctx.sm_scale * 1.4426950408889634)
        _attn_bwd[grid](
            query,
            arg_k,
            value,
            ctx.sm_scale,
            grad_output,
            dq,
            dk,
            dv,
            m,
            delta,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            query.stride(3),
            m.stride(0),
            m.stride(1),
            m.stride(2),
            query.shape[1],
            query.shape[2],
            ctx.original_seq_len,
            BLOCK_M1=BLOCK_M1,
            BLOCK_N1=BLOCK_N1,
            BLOCK_M2=BLOCK_M2,
            BLOCK_N2=BLOCK_N2,
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,
            HEAD_DIM=query.shape[3],
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
            CAUSAL=ctx.causal,
        )

        return (
            dq[:, :, : ctx.original_seq_len, :],
            dk[:, :, : ctx.original_seq_len, :],
            dv[:, :, : ctx.original_seq_len, :],
            None,
            None,
        )


def dense_flash_attention_training(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    sm_scale: float,
    causal: bool,
) -> torch.Tensor:
    return _DenseFlashAttentionTrainingFunction.apply(query, key, value, sm_scale, causal)
