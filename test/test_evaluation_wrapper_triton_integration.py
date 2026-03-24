from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import torch

import triton_qwen3vl


class DummyAccelerator:
    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.device = torch.device("cpu")

    def prepare(self, model):
        return model

    def unwrap_model(self, model):
        return model


class DummyProcessor:
    pass


class DummyLoadedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(visual=SimpleNamespace())
        self.config = SimpleNamespace(use_cache=True, pad_token_id=None, eos_token_id=1)


def test_vlmmodel_init_installs_triton_attention_patch(monkeypatch):
    loaded_model = DummyLoadedModel()
    recorded = {}
    sentinel_state = {
        "installed": True,
        "triton_available": True,
        "mutated_configs": ["vision", "text"],
    }

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = type(
        "AutoProcessor",
        (),
        {"from_pretrained": staticmethod(lambda model_path: DummyProcessor())},
    )
    fake_transformers.AutoModelForImageTextToText = type(
        "AutoModelForImageTextToText",
        (),
        {"from_pretrained": staticmethod(lambda *args, **kwargs: loaded_model)},
    )
    fake_accelerate = types.ModuleType("accelerate")
    fake_accelerate.Accelerator = DummyAccelerator
    fake_accelerate.init_empty_weights = lambda *args, **kwargs: None
    fake_accelerate.load_checkpoint_and_dispatch = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "accelerate", fake_accelerate)
    sys.modules.pop("evaluation_wrapper", None)
    evaluation_wrapper = importlib.import_module("evaluation_wrapper")

    monkeypatch.setattr(evaluation_wrapper.VLMModel, "_explore_model_structure", lambda self: None)
    monkeypatch.setattr(evaluation_wrapper.VLMModel, "_optimize_cross_modal_connector", lambda self: None)

    def fake_install(model):
        recorded["model"] = model
        return sentinel_state

    monkeypatch.setattr(triton_qwen3vl, "install_qwen3vl_triton_attention", fake_install)

    wrapper = evaluation_wrapper.VLMModel("stub-model-path", device="cpu")

    assert recorded["model"] is loaded_model
    assert wrapper.model is loaded_model
    assert wrapper.processor.__class__ is DummyProcessor
    assert wrapper._triton_attention_patch == sentinel_state
    assert "flash_attention" in wrapper._optimizations_applied
