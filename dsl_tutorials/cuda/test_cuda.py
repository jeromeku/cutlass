import cuda.bindings.driver as cuda
from cuda.core.experimental._utils.cuda_utils import handle_return
import numpy as np
from contextlib import contextmanager

DEVICE = 0
driver = cuda

# Start conservative; bump later once it works.
kPTX = r"""
.version 8.0
.target sm_90
.address_size 64

.visible .entry add1(
    .param .u64 arr,
    .param .u32 n
){
    .reg .pred %p;
    .reg .b32  %r<5>;
    .reg .b64  %ptr, %addr;
    .reg .f32  %val;

    ld.param.u64 %ptr, [arr];
    ld.param.u32 %r1,   [n];

    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mad.lo.s32 %r0, %r3, %r4, %r2;

    setp.ge.u32 %p, %r0, %r1;
    @%p bra DONE;

    mul.wide.u32 %addr, %r0, 4;
    add.s64 %addr, %ptr, %addr;

    ld.global.f32 %val, [%addr];
    add.f32 %val, %val, 1.0;
    st.global.f32 [%addr], %val;

DONE:
    ret;
}
"""
def _cudaGetErrorEnum(error):
    if isinstance(error, driver.CUresult):
        err, name = driver.cuGetErrorName(error)
        return name if err == driver.CUresult.CUDA_SUCCESS else "<unknown>"
    # elif isinstance(error, nvrtc.nvrtcResult):
    #     return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError('Unknown error type: {}'.format(error))

def checkCudaErrors(result):
    if result[0].value:
        raise RuntimeError("CUDA error code={}({})".format(result[0].value, _cudaGetErrorEnum(result[0])))
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


def load_module(ptx: str):

    ptx_bytes = ptx.strip().encode("utf-8") + b"\x00"

    info_log = bytearray(16384)
    err_log  = bytearray(16384)

    option_keys = [
        cuda.CUjit_option.CU_JIT_INFO_LOG_BUFFER,
        cuda.CUjit_option.CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        cuda.CUjit_option.CU_JIT_ERROR_LOG_BUFFER,
        cuda.CUjit_option.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        cuda.CUjit_option.CU_JIT_LOG_VERBOSE,
        # cuda.CUjit_option.CU_JIT_TARGET,
    ]
    option_vals = [
        info_log,
        len(info_log),
        err_log,
        len(err_log),
        1,
        # cuda.CUtarget.CU_TARGET_FROM_CUCONTEXT,
    ]

    # ---- Call directly, inspect return + logs ----
    res, mod = cuda.cuModuleLoadDataEx(ptx_bytes,
                                       len(option_keys),
                                       option_keys,
                                       option_vals)
    if res != cuda.CUresult.CUDA_SUCCESS:
        # The logs are C-strings; decode up to the first NUL.
        def cstr(b):
            try:
                n = b.index(0)
            except ValueError:
                n = len(b)
            return b[:n].decode(errors="replace")
        print("JIT INFO:\n", cstr(info_log))
        print("JIT ERRORS:\n", cstr(err_log))
        # Now raise the original error so your test harness still fails.
        raise RuntimeError(f"cuModuleLoadDataEx failed: {res}")
    return mod

def allocate_inputs(num_elements: int = 16, val: int = 1):
    N = np.array(num_elements, dtype=np.uint32)
    hX = np.full(N, val, dtype=np.float32)
    # hX = np.ones(N, dtype=np.float32)

    BUFFER_SIZE = N * hX.itemsize
    assert BUFFER_SIZE == hX.nbytes

    _dX = checkCudaErrors(cuda.cuMemAlloc(hX.nbytes))
    checkCudaErrors(cuda.cuMemcpyHtoD(_dX, hX.ctypes.data, hX.nbytes))
    dX = np.array([int(_dX)], dtype=np.int64)

    return hX, dX, N

def prepare_kernel_params(*args):
    return np.array([arg.ctypes.data for arg in args], dtype=np.uint64)

def test_cuda_ptx():
    checkCudaErrors(cuda.cuInit(0))
    dev = checkCudaErrors(cuda.cuDeviceGet(DEVICE))
    # ctx = handle_return(cuda.cuDevicePrimaryCtxRetain(DEVICE))
    ctx = checkCudaErrors(cuda.cuCtxCreate(0, dev))
    checkCudaErrors(cuda.cuCtxSetCurrent(ctx))

    mod = load_module(kPTX)

    func = handle_return(cuda.cuModuleGetFunction(mod, b"add1"))
    
    # kernel params are pointers to host values
    VAL = 1
    hX, dX, N = allocate_inputs(num_elements=16, val=VAL)
    kernel_params = prepare_kernel_params(dX, N)

    # cuda-python wants a tuple of pointers; Helper marshals them
    checkCudaErrors(cuda.cuLaunchKernel(
        func,
        1, 1, 1,          # grid
        int(N), 1, 1,     # block (one thread per elem for demo)
        0, #sharedmem 
        0, # stream
        kernel_params.ctypes.data,    # kernel params
        0
    ))
    checkCudaErrors(cuda.cuCtxSynchronize())
    
    out = np.empty_like(hX)
    checkCudaErrors(cuda.cuMemcpyDtoH(out.ctypes.data, dX, out.nbytes))
    print("out:", out[:5])
    assert np.allclose(out, np.full_like(hX, VAL + 1))
    
    checkCudaErrors(cuda.cuMemFree(dX.item()))
    checkCudaErrors(cuda.cuModuleUnload(mod))
    checkCudaErrors(cuda.cuCtxDestroy(ctx))

test_cuda_ptx()