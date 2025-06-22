import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass.cute import Float32, Uint64
import torch
import cuda.bindings.driver as cuda

@dsl_user_op
def set_block_rank(
    smem_ptr: cute.Pointer, peer_cta_rank_in_cluster: cute.Int32, *, loc=None, ip=None
) -> cutlass.Int32:
    """Map the given smem pointer to the address at another CTA rank in the cluster."""
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_cta_rank_in_cluster.ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def log2f(a: float | Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "lg2.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def add_op(a: cutlass.Numeric, b: cutlass.Numeric, loc=None, ip=None):

    ptx = """
add.u32 $0, $1, $2;
add.u32 $0, $0, $0;
"""
    return cutlass.Int32(llvm.inline_asm(
        T.i32(),
        [a.ir_value(loc=loc,ip=ip),b.ir_value(loc=loc,ip=ip)],
        ptx,
        "=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    ))

"""
    asm volatile(
        "{\n\t"
        "  .reg .u64 start, target, now;\n\t"
        "  .reg .pred cond;\n\t"
        "  mov.u64 start, %%clock64;\n\t"
        "  add.u64 target, start, %0;\n\t"
        "Loop:\n\t"
        "  mov.u64  now, %%clock64;\n\t"
        "  setp.lt.u64 cond, now, target;\n\t"
        "  @cond bra Loop;\n\t"
        "}"
        :
        : "l"(CYCLES * delay)
        : "memory");

"""

@dsl_user_op
def _spin_kernel(loc=None, ip=None):

    ptx = 'mov.u32 $0, %clock;'
    start = llvm.inline_asm(
        T.i32(),
        [],
        ptx,
        "=r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    out = cutlass.Int32(start)
    cute.printf("Start time: {}", out)


@dsl_user_op
def global_timer_kernel(loc=None, ip=None):

    ptx = 'mov.u32 $0, %globaltimer_lo;'
    start = llvm.inline_asm(
        T.i32(),
        [],
        ptx,
        "=r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    out = cutlass.Int32(start)
    cute.printf("Start time: {}", out)

NUM_CYCLES = cute.Uint32(1_000_000)

@dsl_user_op
def spin_kernel(loc=None, ip=None):
    delay = cute.Uint32(1_000_000)
    ptx = """
    .reg .u32 start, target, now;
    .reg .pred cond;
    mov.u32 start, %globaltimer_lo;
    add.u32 $0, start, $1;
"""
    stop_time = llvm.inline_asm(
        T.i32(),
        [delay.ir_value(loc=loc, ip=ip)],
        ptx,
        "=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    stop = cutlass.Int32(stop_time)
    cute.printf("Stop time: {}", stop)
    # out = cutlass.Int32(start)
    # cute.printf("Start time: {}", out)

@cute.kernel
def inline_add_kernel():
    cute.printf("Hello")
    out = add_op(cute.Int32(1), cute.Int32(2))
    cute.printf("Add op: {}", out)

@cute.kernel
def inline_spin_kernel():
    cute.printf("Spin kernel")
    spin_kernel()


@cute.kernel
def inline_global_timer():
    cute.printf("Global Timer")
    global_timer_kernel()

@cute.jit
def launcher():
    kernel = inline_spin_kernel()
    kernel.launch(grid=(1,1,1), block=(1,1,1))

if __name__ == "__main__":
    a = torch.empty(1, device="cuda", dtype=torch.uint8)
    # a_ = from_dlpack(a, assumed_align=16)
    
    launcher()