import cuda.bindings.driver as cuda
from cuda.core.experimental._utils.cuda_utils import handle_return

DEVICE = 0

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
    .reg .b32  %r, %N, %tid, %bid, %bdim;
    .reg .b64  %ptr, %addr;
    .reg .f32  %val;

    ld.param.u64 %ptr, [arr];
    ld.param.u32 %N,   [n];

    mov.u32 %tid, %tid.x;
    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mad.lo.s32 %r, %bid, %bdim, %tid;

    setp.ge.u32 %p, %r, %N;
    @%p bra DONE;

    mul.wide.u32 %addr, %r, 4;
    add.s64 %addr, %ptr, %addr;

    ld.global.f32 %val, [%addr];
    add.f32 %val, %val, 1.0;
    st.global.f32 [%addr], %val;

DONE:
    ret;
}
"""

def test_cuda_ptx():
    handle_return(cuda.cuInit(0))
    _err, dev = cuda.cuDeviceGet(DEVICE)
    ctx = handle_return(cuda.cuDevicePrimaryCtxRetain(DEVICE))
    handle_return(cuda.cuCtxSetCurrent(ctx))

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

    ptx_bytes = kPTX.strip().encode("utf-8") + b"\x00"

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

    print("Module loaded:", mod)
test_cuda_ptx()