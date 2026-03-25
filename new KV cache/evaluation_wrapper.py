"""
AICAS 2026 - Participant Core Modification File

Participants should modify the VLMModel class to implement optimizations.

Note:
- Benchmark directly calls self.model.generate() for performance testing.
- Your optimizations should modify self.model or its operators in __init__ via Monkey Patch.
- The generate() method is optional and mainly for debugging.
"""
import inspect
import types
from typing import Dict
try:
    from PIL import Image
except ImportError:
    # For testing without PIL
    class Image:
        pass
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat KV heads to match attention head count.
    """
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def _pack_2bit(values: torch.Tensor) -> tuple[torch.Tensor, int]:
    """
    Pack uint8 values in [0, 3] into uint8 bytes, 4 values per byte.
    """
    original_last_dim = values.shape[-1]
    pad = (-original_last_dim) % 4
    if pad > 0:
        values = torch.nn.functional.pad(values, (0, pad))

    packed = (
        values[..., 0::4]
        | (values[..., 1::4] << 2)
        | (values[..., 2::4] << 4)
        | (values[..., 3::4] << 6)
    )
    return packed.contiguous(), original_last_dim


def _unpack_2bit(packed: torch.Tensor, original_last_dim: int) -> torch.Tensor:
    unpacked = torch.stack(
        [
            packed & 0x03,
            (packed >> 2) & 0x03,
            (packed >> 4) & 0x03,
            (packed >> 6) & 0x03,
        ],
        dim=-1,
    ).reshape(*packed.shape[:-1], -1)
    return unpacked[..., :original_last_dim].contiguous()


def _quantize_kivi_key_chunk(key_chunk: torch.Tensor, group_size: int) -> dict:
    """
    KIVI-style key quantization: per-channel over sequence groups.
    Input shape: [B, KVH, S, D]
    """
    batch, kv_heads, seq_len, head_dim = key_chunk.shape
    transposed = key_chunk.permute(0, 1, 3, 2).contiguous()
    seq_pad = (-seq_len) % group_size
    if seq_pad > 0:
        transposed = torch.nn.functional.pad(transposed, (0, seq_pad))
    grouped = transposed.view(batch, kv_heads, head_dim, -1, group_size)

    min_vals = grouped.amin(dim=-1, keepdim=True)
    max_vals = grouped.amax(dim=-1, keepdim=True)
    scales = (max_vals - min_vals).clamp_min(1e-5) / 3.0
    quantized = ((grouped - min_vals) / scales).round().clamp(0, 3).to(torch.uint8)
    packed, packed_last_dim = _pack_2bit(quantized)
    return {
        "packed": packed,
        "min": min_vals.to(torch.float16).contiguous(),
        "scale": scales.to(torch.float16).contiguous(),
        "seq_len": seq_len,
        "group_size": group_size,
        "packed_last_dim": packed_last_dim,
    }


def _dequantize_kivi_key_chunk(quantized: dict, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    packed = quantized["packed"].to(device=device, non_blocking=True)
    min_vals = quantized["min"].to(device=device, dtype=dtype, non_blocking=True)
    scales = quantized["scale"].to(device=device, dtype=dtype, non_blocking=True)
    unpacked = _unpack_2bit(packed, quantized["packed_last_dim"]).to(dtype)
    unpacked = unpacked.view(*min_vals.shape[:-1], quantized["group_size"])
    restored = unpacked * scales + min_vals
    restored = restored.view(*restored.shape[:-2], -1)[..., : quantized["seq_len"]]
    return restored.permute(0, 1, 3, 2).contiguous()


def _quantize_kivi_value_chunk(value_chunk: torch.Tensor, group_size: int) -> dict:
    """
    KIVI-style value quantization: per-token over head-dim groups.
    Input shape: [B, KVH, S, D]
    """
    batch, kv_heads, seq_len, head_dim = value_chunk.shape
    dim_pad = (-head_dim) % group_size
    if dim_pad > 0:
        value_chunk = torch.nn.functional.pad(value_chunk, (0, dim_pad))
    grouped = value_chunk.view(batch, kv_heads, seq_len, -1, group_size)

    min_vals = grouped.amin(dim=-1, keepdim=True)
    max_vals = grouped.amax(dim=-1, keepdim=True)
    scales = (max_vals - min_vals).clamp_min(1e-5) / 3.0
    quantized = ((grouped - min_vals) / scales).round().clamp(0, 3).to(torch.uint8)
    packed, packed_last_dim = _pack_2bit(quantized)
    return {
        "packed": packed,
        "min": min_vals.to(torch.float16).contiguous(),
        "scale": scales.to(torch.float16).contiguous(),
        "head_dim": head_dim,
        "group_size": group_size,
        "packed_last_dim": packed_last_dim,
    }


def _dequantize_kivi_value_chunk(quantized: dict, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    packed = quantized["packed"].to(device=device, non_blocking=True)
    min_vals = quantized["min"].to(device=device, dtype=dtype, non_blocking=True)
    scales = quantized["scale"].to(device=device, dtype=dtype, non_blocking=True)
    unpacked = _unpack_2bit(packed, quantized["packed_last_dim"]).to(dtype)
    unpacked = unpacked.view(*min_vals.shape[:-1], quantized["group_size"])
    restored = unpacked * scales + min_vals
    restored = restored.view(*restored.shape[:-2], -1)[..., : quantized["head_dim"]]
    return restored.contiguous()


def _append_to_paged_kv_cache(module, key_states: torch.Tensor, value_states: torch.Tensor):
    """
    Keep per-layer KV cache in paged form using:
    - KIVI-style 2-bit quantized pages
    - packed uint8 storage
    - a page table for external fragmentation management
    """
    if getattr(module, "_kv_quant_blocks", None) is None:
        module._kv_quant_blocks = []
        module._kv_page_table = []
        module._kv_page_buffer_key = None
        module._kv_page_buffer_value = None
        module._kv_next_logical_page = 0

    buffer_key = getattr(module, "_kv_page_buffer_key", None)
    buffer_value = getattr(module, "_kv_page_buffer_value", None)
    if buffer_key is None:
        buffer_key = key_states.detach().contiguous()
        buffer_value = value_states.detach().contiguous()
    else:
        buffer_key = torch.cat([buffer_key, key_states.detach()], dim=-2).contiguous()
        buffer_value = torch.cat([buffer_value, value_states.detach()], dim=-2).contiguous()

    group_size = max(1, getattr(module, "_gpu_kv_quant_group_size", 32))
    page_size = max(1, getattr(module, "_gpu_kv_page_size", 64))

    while buffer_key.shape[-2] >= page_size:
        page_key = buffer_key[:, :, :page_size, :].contiguous()
        page_value = buffer_value[:, :, :page_size, :].contiguous()

        quant_key = _quantize_kivi_key_chunk(page_key, group_size)
        quant_value = _quantize_kivi_value_chunk(page_value, group_size)

        block_id = len(module._kv_quant_blocks)
        module._kv_quant_blocks.append(
            {
                "key": quant_key,
                "value": quant_value,
                "valid_tokens": page_size,
            }
        )
        module._kv_page_table.append(
            {
                "logical_page_id": module._kv_next_logical_page,
                "block_id": block_id,
            }
        )
        module._kv_next_logical_page += 1
        buffer_key = buffer_key[:, :, page_size:, :].contiguous()
        buffer_value = buffer_value[:, :, page_size:, :].contiguous()

    module._kv_page_buffer_key = buffer_key
    module._kv_page_buffer_value = buffer_value


def _gather_paged_kv_cache(module):
    quant_blocks = getattr(module, "_kv_quant_blocks", None) or []
    page_table = getattr(module, "_kv_page_table", None) or []
    page_buffer_key = getattr(module, "_kv_page_buffer_key", None)
    page_buffer_value = getattr(module, "_kv_page_buffer_value", None)

    key_tensors = []
    value_tensors = []

    if page_buffer_key is not None:
        device = page_buffer_key.device
        dtype = page_buffer_key.dtype
    elif quant_blocks:
        device = quant_blocks[0]["key"]["packed"].device
        dtype = torch.float16
    else:
        return None, None

    for page_entry in page_table:
        block = quant_blocks[page_entry["block_id"]]
        block_key = _dequantize_kivi_key_chunk(block["key"], device=device, dtype=dtype)
        block_value = _dequantize_kivi_value_chunk(block["value"], device=device, dtype=dtype)
        valid_tokens = block["valid_tokens"]
        if valid_tokens > 0:
            key_tensors.append(block_key[:, :, :valid_tokens, :])
            value_tensors.append(block_value[:, :, :valid_tokens, :])

    if page_buffer_key is not None and page_buffer_value is not None:
        key_tensors.append(page_buffer_key)
        value_tensors.append(page_buffer_value)

    if not key_tensors or not value_tensors:
        return None, None

    return torch.cat(key_tensors, dim=-2), torch.cat(value_tensors, dim=-2)


def _populate_custom_cache_from_original(
    module,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
):
    module._kv_quant_blocks = []
    module._kv_page_table = []
    module._kv_page_buffer_key = None
    module._kv_page_buffer_value = None
    module._kv_next_logical_page = 0
    _append_to_paged_kv_cache(module, key_states, value_states)


def _paged_kv_attention(
    module,
    query_states: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attention_mask: torch.Tensor = None,
):
    """
    Compute attention from paged KV cache stored on GPU.
    """
    key_states = _repeat_kv(key_cache, module.num_key_value_groups)
    value_states = _repeat_kv(value_cache, module.num_key_value_groups)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * module.scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
    return attn_output, attn_weights


def _quoka_prefill_attention(
    module,
    query_states: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attention_mask: torch.Tensor = None,
):
    """
    QUOKA-style prefill attention:
    - score KV cache by cosine similarity
    - keep only top-k KV positions for each query position
    """
    key_states = _repeat_kv(key_cache, module.num_key_value_groups)
    value_states = _repeat_kv(value_cache, module.num_key_value_groups)

    normalized_query = torch.nn.functional.normalize(query_states, dim=-1)
    normalized_key = torch.nn.functional.normalize(key_states, dim=-1)
    cosine_scores = torch.matmul(normalized_query, normalized_key.transpose(2, 3))

    total_kv_tokens = cosine_scores.shape[-1]
    top_k = max(1, min(getattr(module, "_prefill_quoka_top_k", 128), total_kv_tokens))
    topk_indices = torch.topk(cosine_scores, k=top_k, dim=-1).indices

    expanded_key_states = key_states.unsqueeze(2).expand(-1, -1, query_states.shape[-2], -1, -1)
    expanded_value_states = value_states.unsqueeze(2).expand(-1, -1, query_states.shape[-2], -1, -1)
    gather_index = topk_indices.unsqueeze(-1).expand(-1, -1, -1, -1, key_states.shape[-1])
    selected_key_states = expanded_key_states.gather(3, gather_index)
    selected_value_states = expanded_value_states.gather(3, gather_index)
    selected_scores = torch.gather(cosine_scores, dim=-1, index=topk_indices)

    attn_weights = selected_scores * module.scaling
    if attention_mask is not None:
        if attention_mask.shape[1] == 1 and topk_indices.shape[1] != 1:
            attention_mask = attention_mask.expand(-1, topk_indices.shape[1], -1, -1)
        selected_mask = torch.gather(attention_mask, dim=-1, index=topk_indices)
        attn_weights = attn_weights + selected_mask

    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.sum(attn_weights.unsqueeze(-1) * selected_value_states, dim=-2)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def _build_swiftkv_decoder_layer_forward(original_forward):
    """
    SwiftKV-style approximation for prefill only.

    During prefill, selected decoder layers are skipped to reduce TTFT.
    Decode stage keeps the normal layer path.
    """

    def patched_forward(self, hidden_states, *args, **kwargs):
        # Prefill usually has sequence length > 1, while decode is typically 1 token.
        is_prefill = hidden_states.shape[1] > 1
        if is_prefill and getattr(self, "_swiftkv_skip_in_prefill", False):
            outputs = (hidden_states,)
            if kwargs.get("output_attentions", False):
                outputs += (None,)
            if kwargs.get("use_cache", False):
                outputs += (kwargs.get("past_key_value", None),)
            return outputs
        return original_forward(hidden_states, *args, **kwargs)

    return patched_forward


def _build_paged_kv_forward(original_forward):
    """
    Build a conservative attention forward that applies paged KV cache.

    If the runtime module shape or helper functions differ from expectation,
    we fall back to the original implementation.
    """

    multimodal_rope_fn = original_forward.__globals__.get("apply_multimodal_rotary_pos_emb")
    rope_fn = original_forward.__globals__.get("apply_rotary_pos_emb") or multimodal_rope_fn
    original_signature = inspect.signature(original_forward)

    def _call_original(hidden_states, position_embeddings, attention_mask, past_key_values, kwargs):
        call_kwargs = dict(kwargs)
        call_kwargs.pop("hidden_states", None)
        call_kwargs.pop("position_embeddings", None)
        call_kwargs.pop("attention_mask", None)
        call_kwargs.pop("past_key_values", None)
        call_kwargs.pop("past_key_value", None)
        if "hidden_states" in original_signature.parameters:
            call_kwargs["hidden_states"] = hidden_states
        if "position_embeddings" in original_signature.parameters:
            call_kwargs["position_embeddings"] = position_embeddings
        if "attention_mask" in original_signature.parameters:
            call_kwargs["attention_mask"] = attention_mask
        if "past_key_values" in original_signature.parameters:
            call_kwargs["past_key_values"] = past_key_values
        elif "past_key_value" in original_signature.parameters:
            call_kwargs["past_key_value"] = past_key_values
        return original_forward(**call_kwargs)

    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        if past_key_values is None and "past_key_value" in kwargs:
            past_key_values = kwargs["past_key_value"]

        if rope_fn is None:
            return _call_original(
                hidden_states, position_embeddings, attention_mask, past_key_values, kwargs
            )

        try:
            cos = None
            sin = None
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            query_states = self.q_proj(hidden_states).view(hidden_shape)
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            value_states = self.v_proj(hidden_states).view(hidden_shape)

            if hasattr(self, "q_norm"):
                query_states = self.q_norm(query_states)
            if hasattr(self, "k_norm"):
                key_states = self.k_norm(key_states)

            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            if position_embeddings is not None:
                cos, sin = position_embeddings
                if multimodal_rope_fn is not None and rope_fn is multimodal_rope_fn:
                    rope_scaling = getattr(self, "rope_scaling", None)
                    mrope_section = None
                    if rope_scaling is not None:
                        mrope_section = rope_scaling.get("mrope_section")
                    if mrope_section is None and hasattr(self, "config"):
                        config_rope_scaling = getattr(self.config, "rope_scaling", None)
                        if config_rope_scaling is not None:
                            mrope_section = config_rope_scaling.get("mrope_section")
                    if mrope_section is None:
                        return _call_original(
                            hidden_states, position_embeddings, attention_mask, past_key_values, kwargs
                        )
                    query_states, key_states = rope_fn(
                        query_states,
                        key_states,
                        cos,
                        sin,
                        mrope_section,
                    )
                else:
                    query_states, key_states = rope_fn(query_states, key_states, cos, sin)

            decode_seq_len = query_states.shape[-2]
            start_after_tokens = max(0, getattr(self, "_kv_start_after_tokens", 0))

            if decode_seq_len > 1:
                self._kv_quant_blocks = []
                self._kv_page_table = []
                self._kv_page_buffer_key = None
                self._kv_page_buffer_value = None
                self._kv_next_logical_page = 0
                self._kv_decode_step = 0
                _append_to_paged_kv_cache(self, key_states, value_states)

                key_cache, value_cache = _gather_paged_kv_cache(self)
                if key_cache is None or value_cache is None:
                    return _call_original(
                        hidden_states, position_embeddings, attention_mask, past_key_values, kwargs
                    )

                attn_output, attn_weights = _quoka_prefill_attention(
                    self,
                    query_states,
                    key_cache,
                    value_cache,
                    attention_mask=attention_mask,
                )

                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = self.o_proj(attn_output)
                return attn_output, attn_weights

            if past_key_values is not None and self._kv_decode_step < start_after_tokens:
                self._kv_decode_step += decode_seq_len
                return _call_original(
                    hidden_states, position_embeddings, attention_mask, past_key_values, kwargs
                )

            if past_key_values is not None and self._kv_decode_step == start_after_tokens:
                cache_kwargs = {}
                if cos is not None and sin is not None:
                    cache_kwargs["cos"] = cos
                    cache_kwargs["sin"] = sin
                if "cache_position" in kwargs:
                    cache_kwargs["cache_position"] = kwargs["cache_position"]
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
                _populate_custom_cache_from_original(self, key_states, value_states)
                self._kv_decode_step += decode_seq_len
            else:
                _append_to_paged_kv_cache(self, key_states, value_states)
                self._kv_decode_step += decode_seq_len

            key_cache, value_cache = _gather_paged_kv_cache(self)
            if key_cache is None or value_cache is None:
                return _call_original(
                    hidden_states, position_embeddings, attention_mask, past_key_values, kwargs
                )

            attn_output, attn_weights = _paged_kv_attention(
                self,
                query_states,
                key_cache,
                value_cache,
                attention_mask=attention_mask,
            )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights
        except Exception:
            return _call_original(
                hidden_states, position_embeddings, attention_mask, past_key_values, kwargs
            )

    return patched_forward


def _reset_paged_kv_cache(model):
    text_model = getattr(model, "model", None)
    layers = getattr(text_model, "layers", None)
    if layers is None:
        return

    for layer in layers:
        self_attn = getattr(layer, "self_attn", None)
        if self_attn is None:
            continue
        if hasattr(self_attn, "_kv_quant_blocks"):
            self_attn._kv_quant_blocks = []
        if hasattr(self_attn, "_kv_page_table"):
            self_attn._kv_page_table = []
        if hasattr(self_attn, "_kv_page_buffer_key"):
            self_attn._kv_page_buffer_key = None
        if hasattr(self_attn, "_kv_page_buffer_value"):
            self_attn._kv_page_buffer_value = None
        if hasattr(self_attn, "_kv_next_logical_page"):
            self_attn._kv_next_logical_page = 0
        if hasattr(self_attn, "_kv_decode_step"):
            self_attn._kv_decode_step = 0


def _build_resetting_generate(original_generate):
    """
    Clear paged KV cache before every generation call.
    """

    def patched_generate(self, *args, **kwargs):
        _reset_paged_kv_cache(self)
        return original_generate(*args, **kwargs)

    return patched_generate


class VLMModel:
    """
    Participant optimization class - modify this to implement optimizations.
    
    Optimization Architecture:
    - Split optimizations into separate methods for isolation and testing
    - Enable/disable each optimization independently in __init__
    - Each optimization method can be tested individually
    
    Important Notes:
    1. Benchmark directly calls self.model.generate() for performance testing.
    2. Your optimizations should modify self.model or its operators via Monkey Patch.
    3. All optimizations are applied in __init__ by calling optimization methods.
    """
    
    def __init__(self, model_path: str, device: str = "cuda:0"):
        """
        Initialize model and apply optimizations.
        
        Args:
            model_path: Qwen3-VL-2B-Instruct model path
            device: CUDA device, e.g., "cuda:0"
        """
        self._device = device
        self.model_path = model_path
        
        # Load processor
        print(f"[VLMModel] Loading processor from {model_path}...")
        self._processor = AutoProcessor.from_pretrained(model_path)
        
        # Load model
        print(f"[VLMModel] Loading model with FP16...")
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device
        )
        self._model.eval()
        
        # Track applied optimizations
        self._optimizations_applied = []
        
        # ================================================================
        # Participant Optimization Area - Enable/disable optimizations here
        # Uncomment the optimization methods you want to apply
        # ================================================================
        
        # 1. Vision Encoder Acceleration
        # self._optimize_vision_encoder()
        
        # 2. KV Cache Management
        self._optimize_kv_cache()
        
        # 3. Cross-modal Connector Optimization
        # self._optimize_cross_modal_connector()
        
        # 4. Flash Attention Optimization
        # self._enable_flash_attention()
        
        # 5. Quantization
        # self._apply_quantization()
        
        # Optional: Explore model structure before optimization
        # self._explore_model_structure()
        
        # print(self._model)
        # ================================================================
        
        print(f"[VLMModel] Model loaded successfully on {device}")
        if self._optimizations_applied:
            print(f"[VLMModel] Applied optimizations: {', '.join(self._optimizations_applied)}")
    
    # ================================================================
    # Optimization Methods - Implement your optimizations here
    # ================================================================
    
    def _explore_model_structure(self):
        """
        Helper method to explore model structure.
        
        Use this to understand the model architecture before implementing optimizations.
        This helps identify where to apply monkey patches.
        """
        print("=" * 60)
        print("Model Structure Exploration")
        print("=" * 60)
        
        # Explore vision model structure
        if hasattr(self._model, 'vision_model'):
            print(f"Vision Model: {type(self._model.vision_model)}")
            if hasattr(self._model.vision_model, 'encoder'):
                if hasattr(self._model.vision_model.encoder, 'layers'):
                    print(f"  Vision Encoder Layers: {len(self._model.vision_model.encoder.layers)}")
                    # Show first layer structure
                    if len(self._model.vision_model.encoder.layers) > 0:
                        print(f"  First Layer Type: {type(self._model.vision_model.encoder.layers[0])}")
        else:
            print("Vision Model: Not found (model structure may differ)")
        
        # Explore language model structure
        if hasattr(self._model, 'model'):
            print(f"Language Model: {type(self._model.model)}")
            if hasattr(self._model.model, 'layers'):
                print(f"  Language Model Layers: {len(self._model.model.layers)}")
        else:
            print("Language Model: Not found (model structure may differ)")
        
        # Explore cross-modal components
        cross_modal_attrs = ['connector', 'cross_attn', 'cross_attention', 'proj', 'projector']
        found_components = []
        for attr in cross_modal_attrs:
            if hasattr(self._model, attr):
                found_components.append(attr)
        if found_components:
            print(f"Cross-modal Components: {', '.join(found_components)}")
        else:
            print("Cross-modal Components: Explore manually (structure may vary)")
        
        print("=" * 60)
        print("Tip: Use print(self._model) to see full model structure")
        print("=" * 60)
    
    def _optimize_vision_encoder(self):
        """
        Optimize Vision Encoder for high-resolution image inputs.
        
        Optimization Directions:
        1. Patch embedding convolution optimization
        2. Vision Transformer attention mechanism optimization
        3. Layer normalization optimization
        4. Memory-efficient image processing
        
        Implementation Steps:
        1. Inspect model structure: call self._explore_model_structure()
        2. Identify bottlenecks using profiling tools (PyTorch Profiler, nsys, etc.)
        3. Implement optimized operators (Triton/CUDA kernels)
        4. Replace original operators via monkey patch
        
        Target Components:
        - self._model.vision_model (if exists)
        - Vision encoder layers and attention mechanisms
        - Convolution operations in patch embedding
        """
        # TODO: Implement your Vision Encoder optimization here
        # 
        # Example workflow:
        # 1. from your_optimization import optimized_attention, optimized_conv
        # 2. Inspect: print(self._model.vision_model) to find target layers
        # 3. Replace: layer.self_attn.forward = optimized_attention
        # 4. Test: Run benchmark to verify improvement
        
        if 'vision_encoder' not in self._optimizations_applied:
            self._optimizations_applied.append('vision_encoder')
    
    def _optimize_kv_cache(self):
        """
        Optimize KV Cache management to reduce memory fragmentation.
        
        Optimization Directions:
        1. Memory layout optimization (contiguous memory allocation)
        2. Fragmentation-free allocation strategies
        3. Efficient cache reuse patterns
        4. Dynamic cache sizing
        
        Implementation Steps:
        1. Understand current KV cache implementation in model layers
        2. Design memory-efficient cache allocation strategy
        3. Implement custom KV cache allocator if needed
        4. Apply optimizations via monkey patch or config modification
        
        Target Components:
        - self._model.config (cache configuration)
        - Attention layers (KV cache allocation)
        - Generation loop (cache management)
        """
        # Enable KV Cache first
        self._model.config.use_cache = True
        if hasattr(self._model.config, 'pad_token_id'):
            if self._model.config.pad_token_id is None:
                self._model.config.pad_token_id = self._model.config.eos_token_id
        if hasattr(self._model, "generation_config"):
            self._model.generation_config.use_cache = True

        if not hasattr(self._model, "_original_generate_for_paged_kv"):
            self._model._original_generate_for_paged_kv = self._model.generate
            self._model.generate = types.MethodType(
                _build_resetting_generate(self._model.generate),
                self._model,
            )

        text_model = getattr(self._model, "model", None)
        layers = getattr(text_model, "layers", None)
        patched_layers = 0

        if layers is not None:
            for layer_idx, layer in enumerate(layers):
                self_attn = getattr(layer, "self_attn", None)
                if self_attn is None:
                    continue

                required_attrs = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "head_dim",
                    "num_key_value_heads",
                    "num_key_value_groups",
                    "scaling",
                    "layer_idx",
                ]
                if not all(hasattr(self_attn, attr) for attr in required_attrs):
                    continue

                self_attn._kv_quant_blocks = []
                self_attn._kv_page_table = []
                self_attn._kv_page_buffer_key = None
                self_attn._kv_page_buffer_value = None
                self_attn._kv_next_logical_page = 0
                self_attn._kv_decode_step = 0
                self_attn._kv_start_after_tokens = 4
                self_attn._gpu_kv_page_size = 64
                self_attn._gpu_kv_quant_group_size = 32
                self_attn._prefill_quoka_top_k = 128

                if hasattr(self_attn, "_original_forward_for_paged_kv"):
                    continue

                self_attn._original_forward_for_paged_kv = self_attn.forward
                self_attn.forward = types.MethodType(
                    _build_paged_kv_forward(self_attn.forward),
                    self_attn,
                )
                patched_layers += 1

        if patched_layers > 0:
            print(
                f"[VLMModel] Applied custom paged KIVI-style KV cache "
                f"to {patched_layers} layers "
                f"(page size = 64, group size = 32, "
                f"2-bit quantized KV, packed uint8 storage, page table, "
                f"QUOKA-style prefill top-k attention by cosine similarity, "
                f"custom decoder KV starts after 4 decode tokens)"
            )
        
        if 'kv_cache' not in self._optimizations_applied:
            self._optimizations_applied.append('kv_cache')
    
    def _optimize_cross_modal_connector(self):
        """
        Optimize Cross-modal Connector computation efficiency.
        
        Optimization Directions:
        1. Cross-attention mechanism optimization
        2. Vision-to-language projection optimization
        3. Multi-modal fusion layer efficiency
        4. Feature alignment and transformation optimization
        
        Implementation Steps:
        1. Identify cross-modal components using self._explore_model_structure()
        2. Profile cross-modal operations to find bottlenecks
        3. Implement optimized cross-attention or projection kernels
        4. Replace original operations via monkey patch
        
        Note: Qwen3-VL's cross-modal structure may vary.
        Use model exploration to identify actual component names and locations.
        """
        # TODO: Implement your Cross-modal Connector optimization here
        # 
        # Example workflow:
        # 1. Explore: self._explore_model_structure() to find connector components
        # 2. from your_optimization import optimized_cross_attention
        # 3. Identify: Inspect model to find cross-attention layers
        # 4. Replace: connector.cross_attention.forward = optimized_cross_attention
        # 5. Test: Verify accuracy and performance improvements
        
        if 'cross_modal' not in self._optimizations_applied:
            self._optimizations_applied.append('cross_modal')
    
    def _enable_flash_attention(self):
        """
        Enable or implement Flash Attention optimization.
        
        Implementation Approaches:
        
        Approach 1: Enable PyTorch's Built-in Flash Attention (Simple)
            - Uses torch.backends.cuda.enable_flash_sdp(True)
            - Easy to enable but limited customization
            - May not work for all attention patterns in Qwen3-VL
        
        Approach 2: Implement Custom Flash Attention (Advanced, Recommended)
            - Write custom Triton/CUDA kernels for attention computation
            - Replace torch.nn.functional.scaled_dot_product_attention
            - Full control over attention computation and memory layout
            - Better performance potential but requires more implementation effort
        
        Recommended: Implement Approach 2 for better performance gains.
        Use profiling to identify which attention operations benefit most from optimization.
        """
        # TODO: Choose and implement your Flash Attention approach
        
        # Approach 1: Simple (enable PyTorch built-in)
        # torch.backends.cuda.enable_flash_sdp(True)
        
        # Approach 2: Advanced (custom implementation - recommended)
        # from your_optimization import custom_flash_attention
        # torch.nn.functional.scaled_dot_product_attention = custom_flash_attention
        # 
        # Or replace at layer level:
        # for layer in self._model.model.layers:
        #     layer.self_attn.forward = custom_attention_with_flash
        
        if 'flash_attention' not in self._optimizations_applied:
            self._optimizations_applied.append('flash_attention')
    
    def _apply_quantization(self):
        """
        Apply quantization to reduce model size and speed up inference.
        
        Optimization Directions:
        1. INT8 quantization (8-bit integer)
        2. FP8 quantization (8-bit floating point)
        3. Mixed precision quantization
        4. Dynamic vs static quantization
        
        Implementation Steps:
        1. Choose quantization strategy based on accuracy/performance trade-off
        2. Use quantization libraries (BitsAndBytes, TensorRT, etc.)
        3. Calibrate quantized model on validation data
        4. Verify accuracy preservation
        
        Note: Quantization may require reloading the model with quantization config.
        Consider applying quantization before other optimizations if model reload is needed.
        """
        # TODO: Implement your quantization here
        # 
        # Example workflow:
        # 1. from transformers import BitsAndBytesConfig
        # 2. quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        # 3. Note: May need to reload model with quantization config
        # 4. Test: Verify accuracy and performance improvements
        
        if 'quantization' not in self._optimizations_applied:
            self._optimizations_applied.append('quantization')
    
    # Required properties for benchmark
    @property
    def processor(self):
        """
        Required by benchmark for input processing.
        
        Benchmark uses this to prepare inputs with unified tokenizer.
        """
        return self._processor
    
    @property
    def model(self):
        """
        Required by benchmark for direct model.generate() calls.
        
        Benchmark directly calls self.model.generate() for performance testing.
        Your optimizations should modify this model object or its operators.
        """
        return self._model
    
    @property
    def device(self):
        """
        Required by benchmark for device information.
        """
        return self._device
    
    def generate(
        self, 
        image: Image.Image, 
        question: str, 
        max_new_tokens: int = 128
    ) -> Dict:
        """
        Generate answer (optional method, mainly for debugging).
        
        Note: Benchmark uses self.model.generate() directly for performance testing.
        This method is provided for convenience and debugging purposes.
        
        Args:
            image: PIL Image object
            question: Question text
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            Dict: {
                "text": str,        # Generated text answer
                "token_count": int  # Generated token count
            }
        """
        # Build Qwen3-VL message format
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }]
        
        # Process inputs
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self._device)
        
        # Generate
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                use_cache=True
            )
        
        # Extract generated tokens (remove input part)
        input_len = inputs.input_ids.shape[1]
        generated_ids = output_ids[0][input_len:]
        
        # Decode
        text = self._processor.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        return {
            "text": text,
            "token_count": len(generated_ids)
        }
