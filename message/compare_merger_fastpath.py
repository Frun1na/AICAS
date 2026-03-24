#!/usr/bin/env python3
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText


def fastpath(module, x: torch.Tensor) -> torch.Tensor:
    hidden_size = module.hidden_size
    norm = module.norm

    if module.use_postshuffle_norm:
        x = x.view(-1, hidden_size) if x.is_contiguous() else x.reshape(-1, hidden_size)
        x = F.layer_norm(x, norm.normalized_shape, norm.weight, norm.bias, norm.eps)
    else:
        x = F.layer_norm(x, norm.normalized_shape, norm.weight, norm.bias, norm.eps)
        x = x.view(-1, hidden_size) if x.is_contiguous() else x.reshape(-1, hidden_size)

    x = F.linear(x, module.linear_fc1.weight, module.linear_fc1.bias)
    x = F.gelu(x, approximate="none")
    x = F.linear(x, module.linear_fc2.weight, module.linear_fc2.bias)
    return x


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    model_path = root / "Qwen3-VL-2B-Instruct"

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()

    visual = model.model.visual
    modules = [("merger", visual.merger)] + [
        (f"deepstack_merger_list.{idx}", module)
        for idx, module in enumerate(visual.deepstack_merger_list)
    ]

    for name, module in modules:
        x = torch.randn(2688, 1024, device=next(module.parameters()).device, dtype=torch.float16)
        with torch.inference_mode():
            ref = module.forward(x)
            out = fastpath(module, x)
        max_abs_diff = (ref - out).abs().max().item()
        print(f"{name}: max_abs_diff={max_abs_diff}")


if __name__ == "__main__":
    main()
