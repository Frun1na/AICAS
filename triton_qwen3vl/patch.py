from __future__ import annotations

from typing import Any

from .backends import TEXT_BACKEND_KEY, VISION_BACKEND_KEY, text_triton_attention, vision_triton_attention
from .dense_flash_attention import TRITON_AVAILABLE


def _load_attention_registries():
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    return ALL_ATTENTION_FUNCTIONS, ALL_MASK_ATTENTION_FUNCTIONS


def _set_attr_if_present(target: Any, attr_name: str, value: str) -> bool:
    if target is None or not hasattr(target, attr_name):
        return False
    setattr(target, attr_name, value)
    return True


def install_qwen3vl_triton_attention(model: Any) -> dict[str, Any]:
    attention_registry, mask_registry = _load_attention_registries()

    if VISION_BACKEND_KEY not in attention_registry:
        attention_registry.register(VISION_BACKEND_KEY, vision_triton_attention)
    if TEXT_BACKEND_KEY not in attention_registry:
        attention_registry.register(TEXT_BACKEND_KEY, text_triton_attention)

    if VISION_BACKEND_KEY not in mask_registry:
        mask_registry.register(VISION_BACKEND_KEY, mask_registry["flash_attention_2"])
    if TEXT_BACKEND_KEY not in mask_registry:
        mask_registry.register(TEXT_BACKEND_KEY, mask_registry["flash_attention_2"])

    base_model = getattr(model, "model", None)
    visual = getattr(base_model, "visual", None)
    language_model = getattr(base_model, "language_model", None)
    model_config = getattr(model, "config", None)

    mutated = {
        "model.model.visual.config": _set_attr_if_present(getattr(visual, "config", None), "_attn_implementation", VISION_BACKEND_KEY),
        "model.model.language_model.config": _set_attr_if_present(
            getattr(language_model, "config", None), "_attn_implementation", TEXT_BACKEND_KEY
        ),
        "model.config.vision_config": _set_attr_if_present(
            getattr(model_config, "vision_config", None), "_attn_implementation", VISION_BACKEND_KEY
        ),
        "model.config.text_config": _set_attr_if_present(
            getattr(model_config, "text_config", None), "_attn_implementation", TEXT_BACKEND_KEY
        ),
    }

    return {
        "installed": True,
        "triton_available": TRITON_AVAILABLE,
        "vision_backend_key": VISION_BACKEND_KEY,
        "text_backend_key": TEXT_BACKEND_KEY,
        "vision_varlen_triton_enabled": TRITON_AVAILABLE,
        "text_prefill_gqa_triton_enabled": TRITON_AVAILABLE,
        "text_decode_triton_enabled": TRITON_AVAILABLE,
        "mutated_configs": mutated,
    }
