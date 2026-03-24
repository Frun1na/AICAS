# Quick Start

## 1. 适用范围

当前主线最适合下面这类环境：
- Linux x86_64
- NVIDIA GPU
- CUDA 可用
- Python 3.10
- PyTorch CUDA 版本与本机驱动匹配

推荐：
- Ubuntu 22.04
- PyTorch 2.4.1 + CUDA 12.4 风格环境
- 已安装 `flash-attn`

说明：
- 没有 `flash-attn` 时，代码会回退到 `sdpa`，可以运行，但性能可能变差。
- 没有成功编译 `native_cuda_ops` 时，代码也可以回退到 PyTorch/HF 默认路径，但当前默认优化主线无法完整复现。

## 4. 创建 Python 环境

```bash
python3.10 -m venv .venv
source .venv/bin/activate

pip install -U pip setuptools wheel packaging ninja
```

## 5. 安装 PyTorch

## 6. 安装项目依赖

```bash
pip install -r requirements.txt
pip install huggingface_hub safetensors
```



```bash
pip install triton
pip install flash-attn --no-build-isolation
```

## 7. 编译原生 CUDA 扩展

当前项目的主线优化包含原生 CUDA 扩展 `native_cuda_ops`，需要在目标机器本地编译。

```bash
python setup_native_cuda_ops.py build_ext --inplace
```

编译成功后，做一个检查：

```bash
python - <<'PY'
import native_cuda_ops
print("native_cuda_ops available:", native_cuda_ops.is_available())
PY
```

如果这里输出 `True`，说明原生扩展可用。

如果编译失败，优先检查：
- 是否安装了 CUDA Toolkit
- `nvcc` 是否在 `PATH` 中
- PyTorch 是否为 CUDA 版
- 当前机器是否为 Linux x86_64

## 8. 准备模型和数据

目标机器上需要有下面两个目录：

```text
./Qwen3-VL-2B-Instruct/
./data/
```

其中：
- `Qwen3-VL-2B-Instruct/` 放模型权重
- `data/` 放 benchmark 使用的数据集

目录结构最少应满足：

```text
AICASGCV1/
├── benchmark.py
├── evaluation_wrapper.py
├── Qwen3-VL-2B-Instruct/
└── data/
```

如果模型还没准备，可以参考仓库 README 中的 Hugging Face 下载方式。

## 9. 测试

先跑一个很小的样本数，确认环境没问题：

```bash
python benchmark.py \
  --model-path ./Qwen3-VL-2B-Instruct \
  --dataset-path ./data \
  --output result_smoke.json \
  --num-samples 3
```

最后再跑更完整的测试：

```bash
python benchmark.py \
  --model-path ./Qwen3-VL-2B-Instruct \
  --dataset-path ./data \
  --output result_100.json \
  --num-samples 100
```

