"""
AICAS 2026 - Participant Core Modification File

Participants should modify the VLMModel class to implement optimizations.

Note:
- Benchmark directly calls self.model.generate() for performance testing.
- Your optimizations should modify self.model or its operators in __init__ via Monkey Patch.
- The generate() method is optional and mainly for debugging.
"""
from typing import Dict
import shutil
try:
    from PIL import Image
except ImportError:
    # For testing without PIL
    class Image:
        pass
import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoProcessor
from accelerate import Accelerator, init_empty_weights, load_checkpoint_and_dispatch
import warnings

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
        self.accelerator = Accelerator(mixed_precision="fp16", device_placement=True)

        self._device = self.accelerator.device
        self.model_path = model_path
        
        # Load processor
        print(f"[VLMModel] Loading processor from {model_path}...")
        self._processor = AutoProcessor.from_pretrained(model_path)
        
        # Load model
        print(f"[VLMModel] Loading model with FP16...")
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map='auto'
        )
        self._model = self.accelerator.prepare(self._model)

        self._model.eval()
        
        # Track applied optimizations
        self._optimizations_applied = []
        self._explore_model_structure()  # Optional: Explore model structure before applying optimizations
        print(self._model)
        
        # ================================================================
        # Participant Optimization Area - Enable/disable optimizations here
        # Uncomment the optimization methods you want to apply
        # ================================================================
        
        # 1. Vision Encoder Acceleration
        # self._optimize_vision_encoder()
        
        # 2. KV Cache Management
        # self._optimize_kv_cache()
        
        # 3. Cross-modal Connector Optimization
        self._optimize_cross_modal_connector()
        
        # 4. Flash Attention Optimization
        self._enable_flash_attention()
        
        # 5. Quantization
        # self._apply_quantization()
        
        # Optional: Explore model structure before optimization
        # self._explore_model_structure()
        
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
        
        # TODO: Implement advanced KV Cache optimizations here
        # 
        # Example workflow:
        # 1. from your_optimization import FragmentationFreeKVCache
        # 2. for layer in self._model.model.layers:
        # 3.     layer.attention.custom_kv_cache = FragmentationFreeKVCache()
        # 4. Test: Monitor memory usage and generation speed
        
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
        base_model = self.accelerator.unwrap_model(self._model)
        visual_model = getattr(getattr(base_model, "model", None), "visual", None)
        if visual_model is None:
            warnings.warn(
                "[VLMModel] Cross-modal connector optimization skipped: visual module not found."
            )
            return

        merger_modules = []
        if hasattr(visual_model, "merger"):
            merger_modules.append(("merger", visual_model.merger))

        if hasattr(visual_model, "deepstack_merger_list"):
            for idx, module in enumerate(visual_model.deepstack_merger_list):
                merger_modules.append((f"deepstack_merger_list.{idx}", module))

        if not merger_modules:
            warnings.warn(
                "[VLMModel] Cross-modal connector optimization skipped: no merger modules found."
            )
            return

        compile_fn = getattr(torch, "compile", None)
        has_native_compiler = any(
            shutil.which(compiler_name) is not None
            for compiler_name in ("gcc", "clang", "cc")
        )
        self._cross_modal_connector_kernels = {}

        def build_fastpath(module):
            hidden_size = module.hidden_size
            norm = module.norm
            linear_fc1 = module.linear_fc1
            linear_fc2 = module.linear_fc2
            use_postshuffle_norm = bool(getattr(module, "use_postshuffle_norm", False))

            def fastpath(x: torch.Tensor) -> torch.Tensor:
                if use_postshuffle_norm:
                    x = x.view(-1, hidden_size) if x.is_contiguous() else x.reshape(-1, hidden_size)
                    x = F.layer_norm(
                        x,
                        norm.normalized_shape,
                        norm.weight,
                        norm.bias,
                        norm.eps,
                    )
                else:
                    x = F.layer_norm(
                        x,
                        norm.normalized_shape,
                        norm.weight,
                        norm.bias,
                        norm.eps,
                    )
                    x = x.view(-1, hidden_size) if x.is_contiguous() else x.reshape(-1, hidden_size)

                x = F.linear(x, linear_fc1.weight, linear_fc1.bias)
                x = F.gelu(x, approximate="none")
                x = F.linear(x, linear_fc2.weight, linear_fc2.bias)
                return x

            return fastpath

        for name, module in merger_modules:
            fastpath = build_fastpath(module)
            optimized_forward = fastpath

            if callable(compile_fn) and has_native_compiler:
                try:
                    optimized_forward = compile_fn(
                        fastpath,
                        fullgraph=False,
                        dynamic=True,
                        mode="max-autotune-no-cudagraphs",
                    )
                except Exception as exc:
                    warnings.warn(
                        f"[VLMModel] torch.compile failed for {name}, using eager fastpath instead: {exc}"
                    )
                    optimized_forward = fastpath

            original_forward = module.forward
            state = {
                "optimized_forward": optimized_forward,
                "fallback_forward": fastpath,
                "original_forward": original_forward,
                "name": name,
                "warned": False,
            }

            def patched_forward(
                x: torch.Tensor,
                _state=state,
            ) -> torch.Tensor:
                try:
                    return _state["optimized_forward"](x)
                except Exception as exc:
                    if _state["optimized_forward"] is not _state["fallback_forward"]:
                        _state["optimized_forward"] = _state["fallback_forward"]
                        if not _state["warned"]:
                            warnings.warn(
                                f"[VLMModel] Compiled cross-modal fastpath failed for {_state['name']}, "
                                f"falling back to eager fastpath: {exc}"
                            )
                            _state["warned"] = True
                        return _state["optimized_forward"](x)

                    if not _state["warned"]:
                        warnings.warn(
                            f"[VLMModel] Eager cross-modal fastpath failed for {_state['name']}, "
                            f"falling back to the original forward: {exc}"
                        )
                        _state["warned"] = True
                    return _state["original_forward"](x)

            module.forward = patched_forward
            self._cross_modal_connector_kernels[name] = state
        
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
        base_model = self.accelerator.unwrap_model(self._model)
        try:
            from triton_qwen3vl import install_qwen3vl_triton_attention
        except Exception as exc:
            warnings.warn(
                f"[VLMModel] Triton attention patch skipped: failed to import runtime package: {exc}"
            )
            return

        try:
            patch_state = install_qwen3vl_triton_attention(base_model)
        except Exception as exc:
            warnings.warn(
                f"[VLMModel] Triton attention patch skipped: failed to install attention backends: {exc}"
            )
            return

        self._triton_attention_patch = patch_state

        if not patch_state.get("triton_available", False):
            warnings.warn(
                "[VLMModel] Triton is not available in the current environment; installed fallback-only attention patch."
            )

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
        )

        inputs = self.accelerator.prepare(inputs)
        
        # Generate
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                use_cache=True,
                pad_token_id=self.processor.tokenizer.pad_token_id
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
