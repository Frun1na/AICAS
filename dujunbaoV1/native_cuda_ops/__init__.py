import torch

try:
    from . import _decode_q1_gqa as _ext
    HAS_NATIVE_DECODE_ATTN = True
    IMPORT_ERROR = None
except Exception as exc:
    _ext = None
    HAS_NATIVE_DECODE_ATTN = False
    IMPORT_ERROR = exc


def is_available() -> bool:
    return HAS_NATIVE_DECODE_ATTN


def decode_q1_gqa_forward(query, key_cache, value_cache, visible_len: int, softmax_scale: float):
    if _ext is None:
        raise RuntimeError(f"native decode attention extension is unavailable: {IMPORT_ERROR}")
    return _ext.decode_q1_gqa_forward(query, key_cache, value_cache, visible_len, softmax_scale)


def rmsnorm_forward(input_tensor, weight, eps: float):
    if _ext is None:
        raise RuntimeError(f"native decode attention extension is unavailable: {IMPORT_ERROR}")
    return _ext.rmsnorm_forward(input_tensor, weight, eps)


def silu_mul_forward(gate, up):
    if _ext is None:
        raise RuntimeError(f"native decode attention extension is unavailable: {IMPORT_ERROR}")
    return _ext.silu_mul_forward(gate, up)


def gate_up_silu_forward(input_tensor, gate_weight, up_weight):
    if _ext is None:
        raise RuntimeError(f"native decode attention extension is unavailable: {IMPORT_ERROR}")
    return _ext.gate_up_silu_forward(input_tensor, gate_weight, up_weight)


def dual_linear_forward(input_tensor, weight0, weight1):
    if _ext is None:
        raise RuntimeError(f"native decode attention extension is unavailable: {IMPORT_ERROR}")
    return _ext.dual_linear_forward(input_tensor, weight0, weight1)


def qkv_linear_forward(input_tensor, q_weight, k_weight, v_weight):
    if _ext is None:
        raise RuntimeError(f"native decode attention extension is unavailable: {IMPORT_ERROR}")
    return _ext.qkv_linear_forward(input_tensor, q_weight, k_weight, v_weight)


def qkv_linear_qk_norm_forward(
    input_tensor,
    q_weight,
    k_weight,
    v_weight,
    q_norm_weight,
    k_norm_weight,
    eps: float,
    head_dim: int,
):
    if _ext is None:
        raise RuntimeError(f"native decode attention extension is unavailable: {IMPORT_ERROR}")
    return _ext.qkv_linear_qk_norm_forward(
        input_tensor,
        q_weight,
        k_weight,
        v_weight,
        q_norm_weight,
        k_norm_weight,
        eps,
        head_dim,
    )


def packed_qkv_qk_norm_rope_forward(
    packed_qkv,
    q_out_features: int,
    k_out_features: int,
    v_out_features: int,
    q_norm_weight,
    k_norm_weight,
    cos,
    sin,
    eps: float,
    head_dim: int,
):
    if _ext is None:
        raise RuntimeError(f"native decode attention extension is unavailable: {IMPORT_ERROR}")
    return _ext.packed_qkv_qk_norm_rope_forward(
        packed_qkv,
        q_out_features,
        k_out_features,
        v_out_features,
        q_norm_weight,
        k_norm_weight,
        cos,
        sin,
        eps,
        head_dim,
    )


def lm_head_argmax_forward(hidden_states, weight):
    if _ext is None:
        raise RuntimeError(f"native decode attention extension is unavailable: {IMPORT_ERROR}")
    return _ext.lm_head_argmax_forward(hidden_states, weight)
