from pathlib import Path
import os
import torch
from torch.utils.cpp_extension import load
DEFAULT_CUTLASS_PATH = "/home/jeromeku/cutlass"
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"

def build_groupedmm_ext(
    name="groupedmm_ext",
    src_root=None,
    cutlass_path="/home/jeromeku/cutlass",
    verbose=True,
    build_dir="build",
):
    src_root = Path(src_root or __file__).parent.resolve()
    build_dir = Path(build_dir).absolute().as_posix()
    os.makedirs(build_dir, exist_ok=True)

    incs = ["csrc",
            os.path.join(cutlass_path, "include"),
            os.path.join(cutlass_path, "tools", "util", "include")]

    incflags = [f"-I{p}" for p in incs]

    sources = [
         "csrc/bindings.cpp",
         "csrc/GroupMM.cu"
    ]

    extra_cxx  = ["-O3", "-std=c++17"]
    extra_nvcc = [
        "-O3", "-std=c++17",
        "--expt-relaxed-constexpr", "--expt-extended-lambda",
        "-gencode=arch=compute_90a,code=sm_90a",
    ]
    extra_cxx += incflags
    extra_nvcc += incflags
    
    mod = load(
        name=name,
        sources=sources,
        extra_cflags=extra_cxx,
        extra_cuda_cflags=extra_nvcc,
        # extra_include_paths=incs,            # absolute
        with_cuda=True,
        verbose=verbose,
        build_directory=str(build_dir),      # absolute
        keep_intermediates=True,
    )
    return mod

# Optional convenience shim that allocates out automatically in Python
def grouped_mm(a, b, offs=None, bias=None, mod=None):
    if mod is None:
        mod = build_groupedmm_ext()
    return torch.ops.groupedmm_ext._grouped_mm(a, b, offs, bias)

def grouped_mm_out(a, b, offs, bias, out, mod=None):
    if mod is None:
        mod = build_groupedmm_ext()
    return torch.ops.groupedmm_ext._grouped_mm_out(a, b, offs, bias, out)
