from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": [
        "-O3",
        "-std=c++17",
        "--use_fast_math",
        "-lineinfo",
    ],
}

setup(
    name="native_cuda_ops",
    packages=find_packages(include=["native_cuda_ops", "native_cuda_ops.*"]),
    ext_modules=[
        CUDAExtension(
            name="native_cuda_ops._decode_q1_gqa",
            sources=[
                "native_cuda_ops/decode_q1_gqa.cpp",
                "native_cuda_ops/decode_q1_gqa_kernel.cu",
            ],
            libraries=["cublas"],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
