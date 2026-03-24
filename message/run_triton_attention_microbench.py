#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _maybe_add_project_root() -> None:
    import sys

    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_maybe_add_project_root()

from triton_qwen3vl.fallback import repeat_kv
from triton_qwen3vl.gqa_decode_attention import gqa_decode_attention
from triton_qwen3vl.gqa_flash_attention import gqa_flash_attention
from triton_qwen3vl.varlen_flash_attention import varlen_flash_attention


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Microbenchmark Triton Qwen3-VL attention kernels against SDPA baselines."
    )
    parser.add_argument("--device", type=str, default="cuda", help="Torch device to benchmark on.")
    parser.add_argument("--dtype", type=str, default="fp16", choices=("fp16", "bf16"), help="Tensor dtype.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument("--num-query-heads", type=int, default=16, help="Number of query heads.")
    parser.add_argument("--num-kv-heads", type=int, default=4, help="Number of key/value heads.")
    parser.add_argument("--head-dim", type=int, default=128, help="Attention head dimension.")
    parser.add_argument("--prefill-seq-len", type=int, default=1024, help="Sequence length for prefill GQA.")
    parser.add_argument("--decode-kv-len", type=int, default=8192, help="Cache length for decode GQA.")
    parser.add_argument(
        "--vision-lengths",
        type=str,
        default="256,384,512",
        help="Comma-separated packed vision sequence lengths.",
    )
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations.")
    parser.add_argument("--iters", type=int, default=100, help="Timed benchmark iterations.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    return parser.parse_args()


def _resolve_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def _device_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise ValueError(f"Expected a CUDA device, got: {device}")
    return torch.cuda.current_device() if device.index is None else int(device.index)


def _measure_ms(fn, *, warmup: int, iters: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1000.0 / iters


def _max_abs_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return float((lhs - rhs).abs().max().item())


def _chunked_varlen_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    cu_seq_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    outputs = []
    for start, end in zip(cu_seq_lens[:-1].tolist(), cu_seq_lens[1:].tolist()):
        outputs.append(
            F.scaled_dot_product_attention(
                query[:, :, start:end],
                key[:, :, start:end],
                value[:, :, start:end],
                is_causal=False,
                scale=scale,
            )
        )
    return torch.cat(outputs, dim=2)


def _build_vision_inputs(
    *,
    lengths: list[int],
    batch_size: int,
    num_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if batch_size != 1:
        raise ValueError("Vision packed microbenchmark currently expects batch_size=1.")

    total_tokens = sum(lengths)
    query = torch.randn(batch_size, num_heads, total_tokens, head_dim, device=device, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    cu_seq_lens = torch.tensor([0, *torch.cumsum(torch.tensor(lengths), dim=0).tolist()], device=device, dtype=torch.int32)
    return query, key, value, cu_seq_lens


def _benchmark_prefill(
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    seq_len: int,
    head_dim: int,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    scale = head_dim ** -0.5
    num_key_value_groups = num_query_heads // num_kv_heads
    query = torch.randn(batch_size, num_query_heads, seq_len, head_dim, device=device, dtype=dtype)
    key = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)
    value = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)

    triton_output = gqa_flash_attention(
        query,
        key,
        value,
        causal=True,
        sm_scale=scale,
        num_key_value_groups=num_key_value_groups,
    )
    reference_output = F.scaled_dot_product_attention(
        query,
        repeat_kv(key, num_key_value_groups),
        repeat_kv(value, num_key_value_groups),
        is_causal=True,
        scale=scale,
    )

    triton_ms = _measure_ms(
        lambda: gqa_flash_attention(
            query,
            key,
            value,
            causal=True,
            sm_scale=scale,
            num_key_value_groups=num_key_value_groups,
        ),
        warmup=warmup,
        iters=iters,
        device=device,
    )
    reference_ms = _measure_ms(
        lambda: F.scaled_dot_product_attention(
            query,
            repeat_kv(key, num_key_value_groups),
            repeat_kv(value, num_key_value_groups),
            is_causal=True,
            scale=scale,
        ),
        warmup=warmup,
        iters=iters,
        device=device,
    )

    return {
        "triton_ms": triton_ms,
        "reference_ms": reference_ms,
        "speedup": reference_ms / triton_ms,
        "max_abs_err": _max_abs_diff(triton_output, reference_output),
    }


def _benchmark_decode(
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    kv_len: int,
    head_dim: int,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    scale = head_dim ** -0.5
    num_key_value_groups = num_query_heads // num_kv_heads
    query = torch.randn(batch_size, num_query_heads, 1, head_dim, device=device, dtype=dtype)
    key = torch.randn(batch_size, num_kv_heads, kv_len, head_dim, device=device, dtype=dtype)
    value = torch.randn(batch_size, num_kv_heads, kv_len, head_dim, device=device, dtype=dtype)

    triton_output = gqa_decode_attention(
        query,
        key,
        value,
        sm_scale=scale,
        num_key_value_groups=num_key_value_groups,
    )
    reference_output = F.scaled_dot_product_attention(
        query,
        repeat_kv(key, num_key_value_groups),
        repeat_kv(value, num_key_value_groups),
        is_causal=False,
        scale=scale,
    )

    triton_ms = _measure_ms(
        lambda: gqa_decode_attention(
            query,
            key,
            value,
            sm_scale=scale,
            num_key_value_groups=num_key_value_groups,
        ),
        warmup=warmup,
        iters=iters,
        device=device,
    )
    reference_ms = _measure_ms(
        lambda: F.scaled_dot_product_attention(
            query,
            repeat_kv(key, num_key_value_groups),
            repeat_kv(value, num_key_value_groups),
            is_causal=False,
            scale=scale,
        ),
        warmup=warmup,
        iters=iters,
        device=device,
    )

    return {
        "triton_ms": triton_ms,
        "reference_ms": reference_ms,
        "speedup": reference_ms / triton_ms,
        "max_abs_err": _max_abs_diff(triton_output, reference_output),
    }


def _benchmark_vision_varlen(
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    num_heads: int,
    lengths: list[int],
    head_dim: int,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    scale = head_dim ** -0.5
    query, key, value, cu_seq_lens = _build_vision_inputs(
        lengths=lengths,
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
    )
    metadata_cache: dict = {}

    triton_output = varlen_flash_attention(
        query,
        key,
        value,
        cu_seq_lens_q=cu_seq_lens,
        cu_seq_lens_k=cu_seq_lens,
        sm_scale=scale,
        max_length_q=max(lengths),
        max_length_k=max(lengths),
        metadata_cache=metadata_cache,
    )
    reference_output = _chunked_varlen_sdpa(
        query,
        key,
        value,
        cu_seq_lens=cu_seq_lens,
        scale=scale,
    )

    triton_ms = _measure_ms(
        lambda: varlen_flash_attention(
            query,
            key,
            value,
            cu_seq_lens_q=cu_seq_lens,
            cu_seq_lens_k=cu_seq_lens,
            sm_scale=scale,
            max_length_q=max(lengths),
            max_length_k=max(lengths),
            metadata_cache=metadata_cache,
        ),
        warmup=warmup,
        iters=iters,
        device=device,
    )
    reference_ms = _measure_ms(
        lambda: _chunked_varlen_sdpa(
            query,
            key,
            value,
            cu_seq_lens=cu_seq_lens,
            scale=scale,
        ),
        warmup=warmup,
        iters=iters,
        device=device,
    )

    return {
        "triton_ms": triton_ms,
        "reference_ms": reference_ms,
        "speedup": reference_ms / triton_ms,
        "max_abs_err": _max_abs_diff(triton_output, reference_output),
    }


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    dtype = _resolve_dtype(args.dtype)
    lengths = [int(item) for item in args.vision_lengths.split(",") if item.strip()]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton microbenchmark.")
    if device.type == "cuda":
        torch.cuda.set_device(_device_index(device))
    if args.num_query_heads % args.num_kv_heads != 0:
        raise ValueError("num-query-heads must be divisible by num-kv-heads.")

    results = {
        "device": str(device),
        "dtype": args.dtype,
        "gpu_name": torch.cuda.get_device_name(_device_index(device)),
        "seed": args.seed,
        "prefill": _benchmark_prefill(
            device=device,
            dtype=dtype,
            batch_size=args.batch_size,
            num_query_heads=args.num_query_heads,
            num_kv_heads=args.num_kv_heads,
            seq_len=args.prefill_seq_len,
            head_dim=args.head_dim,
            warmup=args.warmup,
            iters=args.iters,
        ),
        "decode": _benchmark_decode(
            device=device,
            dtype=dtype,
            batch_size=args.batch_size,
            num_query_heads=args.num_query_heads,
            num_kv_heads=args.num_kv_heads,
            kv_len=args.decode_kv_len,
            head_dim=args.head_dim,
            warmup=args.warmup,
            iters=args.iters,
        ),
        "vision_varlen": _benchmark_vision_varlen(
            device=device,
            dtype=dtype,
            batch_size=args.batch_size,
            num_heads=args.num_query_heads,
            lengths=lengths,
            head_dim=args.head_dim,
            warmup=args.warmup,
            iters=args.iters,
        ),
    }

    payload = json.dumps(results, indent=2)
    print(payload)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
