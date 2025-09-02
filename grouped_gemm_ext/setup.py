import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
DEFAULT_CUTLASS_PATH="/home/jeromeku/cutlass"
CUTLASS_PATH = os.environ.get("CUTLASS_PATH", DEFAULT_CUTLASS_PATH)
if not CUTLASS_PATH:
    raise RuntimeError("Set CUTLASS_PATH to your local CUTLASS 3.x root")

cutlass_includes = [
    os.path.join(CUTLASS_PATH, "include"),                 # cutlass + cute
    os.path.join(CUTLASS_PATH, "tools", "util", "include")
]

ext = CUDAExtension(
    name="groupedmm_ext._C",
    sources=[
        "csrc/bindings.cpp",
        # Upstream file copied verbatim:
        "csrc/GroupMM.cu",
    ],
    include_dirs=[
        "csrc",  # so <ATen/native/cuda/...> resolves to your local copies
        *cutlass_includes,
    ],
    extra_compile_args={
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": [
            "-O3", "-std=c++17",
            "--expt-relaxed-constexpr", "--expt-extended-lambda",
            # Hopper only for now (matches your ask). Add SM100 later.
            "-gencode=arch=compute_90a,code=sm_90a",
        ],
    },
)

setup(
    name="groupedmm_ext",
    version="0.1.0",
    packages=["groupedmm_ext"],
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
