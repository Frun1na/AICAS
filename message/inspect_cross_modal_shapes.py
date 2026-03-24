#!/usr/bin/env python3
import json
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForImageTextToText, AutoProcessor


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    model_path = root / "Qwen3-VL-2B-Instruct"
    dataset_path = root / "data"

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()

    dataset = load_from_disk(str(dataset_path))
    sample = dataset[0]

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": sample["image"]},
            {"type": "text", "text": sample["question"]},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = inputs.to(device)

    visual = model.model.visual
    records = []

    def make_hook(name):
        def hook(_, module_inputs, module_output):
            x = module_inputs[0]
            records.append({
                "module": name,
                "shape": tuple(x.shape),
                "dtype": str(x.dtype),
                "device": str(x.device),
                "is_contiguous": bool(x.is_contiguous()),
                "stride": tuple(x.stride()),
                "output_shape": tuple(module_output.shape),
            })
        return hook

    handles = [visual.merger.register_forward_hook(make_hook("merger"))]
    for idx, module in enumerate(visual.deepstack_merger_list):
        handles.append(module.register_forward_hook(make_hook(f"deepstack_merger_list.{idx}")))

    with torch.inference_mode():
        model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            temperature=0.0,
            use_cache=True,
        )

    for handle in handles:
        handle.remove()

    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
