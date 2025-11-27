# Type-Erased Call Chains: Complete Execution Traces

This document provides step-by-step traces showing how type-erased values flow through the CUTLASS system, from user code to hardware instructions.

## Trace 1: Kernel Instantiation with Type-Erased Types

### User Code

```cpp
// User wants runtime-selectable float format
using ElementA = cutlass::type_erased_dynamic_nv_float4_t;
using ElementB = cutlass::type_erased_dynamic_nv_float4_t;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;

using Gemm = /* CollectiveBuilder instantiation */;
```

### Step 1: Type Alias Resolution

```cpp
// Step 1.1: Expand type_erased_dynamic_nv_float4_t
using ElementA = cutlass::type_erased_dynamic_nv_float4_t;

// Step 1.2: Definition lookup
// [include/cutlass/float_subbyte.h:515]
using type_erased_dynamic_nv_float4_t = nv_float4_t<type_erased_dynamic_float4_t>;
//                                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                                                    Template parameter

// Step 1.3: Trait struct instantiation
// [include/cutlass/float_subbyte.h:506-513]
template <class F4Type>
struct nv_float4_t {
    static_assert(cute::is_same_v<F4Type, cutlass::float_e2m1_t>
                  || cute::is_same_v<F4Type, type_erased_dynamic_float4_t>,
                  "Only float_e2m1_t type_erased_dynamic_float4_t allowed");

    using ScaleFactorType = cutlass::float_ue4m3_t;
    using DataType = F4Type;  // = type_erased_dynamic_float4_t
};

// Step 1.4: Result after expansion
struct ElementA_Expanded {
    using ScaleFactorType = cutlass::float_ue4m3_t;
    using DataType = cutlass::type_erased_dynamic_float4_t;
    //               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //               Union type (not concrete type)
};
```

**Visual Flow**:
```
type_erased_dynamic_nv_float4_t
  │
  ├─[typedef expansion]─→ nv_float4_t<type_erased_dynamic_float4_t>
  │
  ├─[template instantiation]
  │
  └─→ struct {
         ScaleFactorType = float_ue4m3_t;
         DataType = type_erased_dynamic_float4_t;  ← Union!
      }
```

### Step 2: Union Definition Lookup

```cpp
// Step 2.1: What is type_erased_dynamic_float4_t?
// [include/cutlass/float_subbyte.h:461-467]
union type_erased_dynamic_float4_t {
    cutlass::float_e2m1_t e2m1;
    // Future: cutlass::float_e1m2_t e1m2;

    CUTLASS_HOST_DEVICE
    explicit operator cutlass::float_e2m1_t() const {
        return e2m1;
    }
};

// Step 2.2: Size specialization
// [include/cutlass/float_subbyte.h:470-472]
template <>
struct sizeof_bits<type_erased_dynamic_float4_t> {
    static constexpr int value = 4;  // ← 4 bits, not 8
};

// Step 2.3: Memory layout
// Union has same size as largest member:
sizeof(type_erased_dynamic_float4_t) == sizeof(float_e2m1_t) == 1 byte
//                                                                (4 bits used)
```

**Visual Memory Layout**:
```
type_erased_dynamic_float4_t union:
┌────────────────────────────────┐
│  Single byte storage           │
│                                │
│  ┌──────────────────────────┐  │
│  │ float_e2m1_t e2m1        │  │  All members
│  │ ┌─────────────────────┐  │  │  share this
│  │ │ uint8_t storage     │  │  │  memory
│  │ │  Bit: 3 2 1 0       │  │  │
│  │ │      ┌─┬───┬─┐      │  │  │
│  │ │      │S│E E│M│      │  │  │
│  │ │      └─┴───┴─┘      │  │  │
│  │ └─────────────────────┘  │  │
│  └──────────────────────────┘  │
│                                │
│  Future members would          │
│  overlay same memory:          │
│  ┌──────────────────────────┐  │
│  │ [float_e1m2_t e1m2]      │  │
│  │ [Different bit layout]   │  │
│  └──────────────────────────┘  │
│                                │
└────────────────────────────────┘
```

### Step 3: CollectiveBuilder Type Propagation

```cpp
// Step 3.1: CollectiveBuilder receives type-erased ElementA
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100,
    cutlass::arch::OpClassTensorOp,
    ElementA,  // = type_erased_dynamic_nv_float4_t
    LayoutA,
    AlignmentA,
    // ...
>::CollectiveOp;

// Step 3.2: Inside CollectiveBuilder
// [include/cutlass/gemm/collective/builders/sm1xx_common.inl]
template <...>
struct CollectiveBuilder {
    using Element = ElementA;
    //             ^^^^^^^^ Type-erased union

    // Extract DataType from trait struct
    using ElementType = typename Element::DataType;
    //                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //                  = type_erased_dynamic_float4_t (union)

    using ScaleType = typename Element::ScaleFactorType;
    //                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //                = float_ue4m3_t

    // Format detection: Compile-time check
    static constexpr bool is_type_erased =
        cute::is_same_v<ElementType, cutlass::type_erased_dynamic_float4_t>;
    //  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //  TRUE for type-erased types, FALSE for concrete types

    // Cannot determine format at compile time!
    static constexpr auto compile_time_format = to_MXF8F6F4Format<ElementType>();
    //                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //                                          Returns MXF8F6F4Format::INVALID
    //                                          (format unknown at compile time)
};
```

**Compile-Time Type Checks**:
```
┌─────────────────────────────────────────────────────────────┐
│              CollectiveBuilder Compile-Time State            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Element            = type_erased_dynamic_nv_float4_t       │
│  ElementType        = type_erased_dynamic_float4_t (union)  │
│  ScaleType          = float_ue4m3_t                         │
│  is_type_erased     = true                                  │
│  compile_time_format = MXF8F6F4Format::INVALID              │
│                                                             │
│  ⚠️  Format UNKNOWN at compile time!                       │
│      Must use runtime argument                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Step 4: Kernel Signature Generation

```cpp
// Step 4.1: Kernel template instantiation
template <
    class ProblemShape,
    class CollectiveMainloop,
    class CollectiveEpilogue,
    class TileScheduler
>
class GemmUniversal {
    // Kernel receives type-erased pointers:
    using ElementA = typename CollectiveMainloop::ElementA;
    //             = type_erased_dynamic_float4_t (union)

    __global__ static void kernel(
        typename CollectiveMainloop::Params const& mainloop_params,
        // ...
    ) {
        // Inside params:
        // - ElementA* ptr_A  (points to type-erased data)
        // - ElementB* ptr_B
        // - MXF8F6F4Format runtime_format_a  ← Runtime enum!
        // - MXF8F6F4Format runtime_format_b
    }
};

// Step 4.2: Kernel function pointer type
using KernelPtr = void (*)(
    type_erased_dynamic_float4_t*,  // Data pointer (union type)
    float_ue4m3_t*,                 // Scale pointer
    MXF8F6F4Format,                 // Runtime format enum
    // ...
);
```

**Key Point**: Kernel is compiled with type-erased signature, accepts runtime format enum.

## Trace 2: Runtime Format Selection and Kernel Launch

### Step 1: User Specifies Runtime Format

```cpp
// User code: Select format at runtime
int main(int argc, char** argv) {
    // Parse command-line argument or configuration
    std::string format_name = argv[1];  // "e2m1"

    // Convert string to enum
    cute::UMMA::MXF8F6F4Format runtime_format;
    if (format_name == "e2m1") {
        runtime_format = cute::UMMA::MXF8F6F4Format::E2M1;
    }
    // Future: else if (format_name == "e1m2") { ... }

    std::cout << "Selected format: " << to_string(runtime_format) << std::endl;
    // Output: "Selected format: E2M1"

    // ... continue to kernel setup ...
}
```

### Step 2: Arguments Preparation

```cpp
// Step 2.1: Create GEMM arguments
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, L},
    {ptr_A, stride_A, ptr_B, stride_B},
    {{alpha, beta}, ptr_C, stride_C, ptr_D, stride_D}
};

// Step 2.2: Assign scale factor pointers
arguments.mainloop.ptr_SFA = scale_a_ptr;  // float_ue4m3_t*
arguments.mainloop.ptr_SFB = scale_b_ptr;  // float_ue4m3_t*

// Step 2.3: ⭐ KEY STEP: Assign runtime format
arguments.hw_info.runtime_format_a = runtime_format;  // MXF8F6F4Format::E2M1
arguments.hw_info.runtime_format_b = runtime_format;
//                ^^^^^^^^^^^^^^^^
//                This tells kernel how to interpret type-erased data!

// Step 2.4: Initialize GEMM
Gemm gemm;
cutlass::Status status = gemm.initialize(arguments);
```

**Arguments Structure Flow**:
```
Gemm::Arguments
├── ProblemShape {M, N, K, L}
├── mainloop
│   ├── ptr_A: type_erased_dynamic_float4_t*
│   ├── ptr_B: type_erased_dynamic_float4_t*
│   ├── ptr_SFA: float_ue4m3_t*
│   └── ptr_SFB: float_ue4m3_t*
└── hw_info
    ├── runtime_format_a: MXF8F6F4Format::E2M1  ← Runtime dispatch!
    └── runtime_format_b: MXF8F6F4Format::E2M1
```

### Step 3: Kernel Launch

```cpp
// Step 3.1: Launch kernel with runtime format
cutlass::Status status = gemm.run();

// Step 3.2: Inside gemm.run()
template <...>
Status GemmUniversalAdapter::run() {
    // Extract kernel parameters
    auto kernel_params = to_underlying_arguments(args_);

    // Launch kernel with cluster launch API
    return ClusterLauncher::launch(
        grid_dims,
        cluster_dims,
        block_dims,
        smem_size,
        stream,
        reinterpret_cast<void const*>(kernel_ptr),
        reinterpret_cast<void**>(&kernel_params)
    );
}

// Step 3.3: Kernel receives parameters including runtime format
__global__ void gemm_kernel(
    Params params  // Contains runtime_format_a, runtime_format_b
) {
    // params.mainloop.runtime_format_a == MXF8F6F4Format::E2M1
    // Now kernel knows how to interpret type-erased data!
}
```

## Trace 3: Device-Side Runtime Dispatch

### Step 1: Kernel Entry and Threadblock Setup

```cpp
__global__ void gemm_kernel(
    typename CollectiveMainloop::Params const& mainloop_params,
    typename CollectiveEpilogue::Params const& epilogue_params
) {
    // Step 1.1: Extract runtime format
    auto format_a = mainloop_params.runtime_format_a;
    // format_a == MXF8F6F4Format::E2M1 (runtime value)

    // Step 1.2: Setup threadblock
    int thread_idx = threadIdx.x + threadIdx.y * blockDim.x +
                     threadIdx.z * blockDim.x * blockDim.y;

    // Step 1.3: Get cluster and block IDs
    auto cluster_shape = cute::cluster_shape();
    auto block_idx_in_cluster = cute::block_idx_in_cluster();

    // Step 1.4: Instantiate mainloop
    CollectiveMainloop mainloop;

    // ...
}
```

### Step 2: TMA Load with Runtime Format

```cpp
// Inside CollectiveMainloop::load()
template <typename Element>
__device__ void load_tile(
    Element* gmem_ptr,           // type_erased_dynamic_float4_t*
    MXF8F6F4Format runtime_format  // E2M1
) {
    // Step 2.1: Compile-time type check
    constexpr bool is_type_erased =
        cute::is_same_v<Element, type_erased_dynamic_float4_t>;

    if constexpr (is_type_erased) {
        // Step 2.2: Runtime format dispatch
        // [include/cute/arch/copy_sm90_desc.hpp]

        // Determine TMA data type based on runtime format
        CUtensorMapDataType tma_dtype;
        switch (runtime_format) {
            case MXF8F6F4Format::E2M1:
                tma_dtype = CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
                break;
            case MXF8F6F4Format::E3M2:
                tma_dtype = CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN8B;
                break;
            // ... other cases
        }

        // Step 2.3: Issue TMA load with correct data type
        asm volatile(
            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
            " [%0], [%1, {%2, %3}], [%4];"
            :
            : "r"(smem_ptr),
              "l"(tma_desc),
              "r"(coord_x),
              "r"(coord_y),
              "r"(tma_dtype)  // ← Runtime format affects instruction!
        );
    }
}
```

**TMA Instruction Selection**:
```
┌────────────────────────────────────────────────────┐
│         Runtime Format → TMA Data Type             │
├────────────────────────────────────────────────────┤
│                                                    │
│  MXF8F6F4Format::E2M1 (value=5)                   │
│         ↓                                          │
│  CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B             │
│         ↓                                          │
│  cp.async.bulk.tensor.2d with 4-bit elements      │
│                                                    │
│  MXF8F6F4Format::E2M3 (value=3)                   │
│         ↓                                          │
│  CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN8B             │
│         ↓                                          │
│  cp.async.bulk.tensor.2d with 6-bit elements      │
│                                                    │
└────────────────────────────────────────────────────┘
```

### Step 3: MMA Instruction with Runtime Format

```cpp
// Inside CollectiveMainloop::mma()
template <typename ElementA, typename ElementB>
__device__ void compute_tile(
    ElementA const* reg_a,  // type_erased_dynamic_float4_t*
    ElementB const* reg_b,
    MXF8F6F4Format format_a,
    MXF8F6F4Format format_b
) {
    // Step 3.1: Runtime format to PTX instruction
    // [include/cute/arch/mma_sm100_desc.hpp]

    // TCGEN05 MMA descriptor with runtime format
    uint32_t mma_desc = encode_mma_descriptor(
        format_a,  // E2M1
        format_b,  // E2M1
        /* other params */
    );

    // Step 3.2: Issue MMA instruction
    asm volatile(
        "tcgen05.mma.cta_group::2.kind::ab.tiled"
        ".f16.f32.f32.f32"
        ".format::%0"  // ← Runtime format encoded in descriptor!
        " {%1}, {%2}, {%3}, {%4};"
        : "=r"(accum_out)
        : "r"(mma_desc),  // Contains format_a and format_b
          "r"(reg_a),
          "r"(reg_b),
          "r"(accum_in)
    );
}
```

**MMA Descriptor Encoding**:
```
TCGEN05 MMA Descriptor (64-bit):
┌──────────────────────────────────────────────────────────┐
│  Bits 63-56 │ Bits 55-48 │ ... │ Bits 7-0               │
├─────────────┼────────────┼─────┼────────────────────────┤
│  Reserved   │  Format B  │ ... │  Format A              │
│             │            │     │                        │
│             │  E2M1 (5)  │     │  E2M1 (5)              │
│             │     ↑      │     │     ↑                  │
│             │  Runtime!  │     │  Runtime!              │
└──────────────────────────────────────────────────────────┘

Hardware decodes format at execution:
- Format A = 5 → Use E2M1 decoding for operand A
- Format B = 5 → Use E2M1 decoding for operand B
- Mantissa handling, exponent bias, etc. all adjusted
```

## Trace 4: Complete Data Flow (Memory → Compute)

### Step 1: Global Memory (Type-Erased Data)

```
Global Memory (GMEM):
┌────────────────────────────────────────────────────────┐
│  Matrix A Block (32 × 64 elements)                     │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Stored as: type_erased_dynamic_float4_t[]            │
│  (2 elements per byte, tightly packed)                │
│                                                        │
│  Byte 0:  ┌──────┬──────┐                             │
│           │ e0   │ e1   │   4 bits each               │
│           └──────┴──────┘                             │
│  Byte 1:  ┌──────┬──────┐                             │
│           │ e2   │ e3   │                             │
│           └──────┴──────┘                             │
│  ...                                                   │
│                                                        │
│  ⚠️  Format unknown from data itself!                 │
│      Interpretation depends on runtime_format_a        │
│                                                        │
├────────────────────────────────────────────────────────┤
│  Scale Factor Block (per-block scaling)               │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Stored as: float_ue4m3_t[]                           │
│  (1 scale per 32 elements)                            │
│                                                        │
│  scale[0] = 2.0  ← For elements 0-31                  │
│  scale[1] = 4.0  ← For elements 32-63                 │
│  ...                                                   │
│                                                        │
└────────────────────────────────────────────────────────┘

Runtime Context (passed separately):
┌────────────────────────────────────────┐
│  runtime_format_a = MXF8F6F4Format::E2M1│
│                    │                   │
│                    └─→ Tells how to    │
│                        interpret data  │
└────────────────────────────────────────┘
```

### Step 2: TMA Load (GMEM → SMEM)

```cpp
// TMA instruction receives runtime format
cp.async.bulk.tensor.2d(
    smem_ptr,
    gmem_ptr,   // type_erased_dynamic_float4_t*
    format      // MXF8F6F4Format::E2M1
);
```

**Hardware Behavior**:
```
┌────────────────────────────────────────────────────────┐
│                   TMA Hardware Unit                    │
├────────────────────────────────────────────────────────┤
│                                                        │
│  1. Read runtime_format from descriptor               │
│     format = E2M1 (value 5)                           │
│                                                        │
│  2. Configure data path:                              │
│     - Element size: 4 bits                            │
│     - Alignment: 8 bytes                              │
│     - Packing: 2 elements/byte                        │
│                                                        │
│  3. Load bytes from GMEM:                             │
│     [raw bytes, no interpretation]                    │
│                                                        │
│  4. Transfer to SMEM:                                 │
│     [still type-erased, format preserved]             │
│                                                        │
│  ⚠️  No conversion! Just memory transfer              │
│      Format interpretation happens in MMA             │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### Step 3: Shared Memory (Still Type-Erased)

```
Shared Memory (SMEM):
┌────────────────────────────────────────────────────────┐
│  Tile Data (type_erased_dynamic_float4_unpacksmem_t)  │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Memory layout optimized for bank conflict avoidance: │
│                                                        │
│  Thread 0:  e0  e1  e2  e3  ...                       │
│  Thread 1:  e16 e17 e18 e19 ...                       │
│  ...                                                   │
│                                                        │
│  Still type-erased! Format = runtime_format_a         │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### Step 4: SMEM → Register (Type-Erased Load)

```cpp
// Load from SMEM to registers
type_erased_dynamic_float4_t reg_a[4];
cute::copy(smem_tile, reg_a);

// reg_a contains raw 4-bit values
// Interpretation deferred until MMA
```

### Step 5: MMA Computation (Format Interpretation!)

```cpp
// ⭐ KEY MOMENT: Hardware interprets format here
asm volatile(
    "tcgen05.mma.format::%0"
    :
    : "r"(encode_format(runtime_format_a))  // E2M1 encoded in descriptor
);
```

**Hardware MMA Unit**:
```
┌────────────────────────────────────────────────────────┐
│               TCGEN05 MMA Hardware                     │
├────────────────────────────────────────────────────────┤
│                                                        │
│  1. Decode MMA descriptor:                            │
│     format_a = E2M1 (5)                               │
│     format_b = E2M1 (5)                               │
│                                                        │
│  2. Configure decode path for Operand A:              │
│     - 1 sign bit, 2 exp bits, 1 mantissa bit         │
│     - Exp bias = 1                                    │
│     - Denormals supported                             │
│                                                        │
│  3. Decode 4-bit value:                               │
│     Raw bits: 0b0101 (from reg_a)                    │
│          ↓                                            │
│     E2M1 decode:                                      │
│       Sign: 0 (positive)                              │
│       Exp:  10₂ = 2 (biased)                          │
│       Mant: 1₂                                        │
│          ↓                                            │
│     Float value: (+1) × 1.1₂ × 2^(2-1) = 3.0         │
│                                                        │
│  4. Perform FP32 accumulation:                        │
│     accum += (decode(A) × scale_a) × (decode(B) × scale_b) │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### Step 6: Final Result

```
Accumulator Registers (FP32):
┌────────────────────────────────────┐
│  accum[0] = 128.5                 │
│  accum[1] = -64.25                │
│  ...                               │
│                                    │
│  High-precision accumulation       │
│  (FP32 or BF16, depending on arch) │
└────────────────────────────────────┘
```

## Comparison: Static vs. Type-Erased Call Chains

### Static Type Path (Compile-Time Format)

```
User Code:
  using ElementA = nv_float4_t<float_e2m1_t>;
                                ^^^^^^^^^^^^^
                                Concrete type
            ↓
Template Instantiation:
  CollectiveBuilder<float_e2m1_t, ...>
  - compile_time_format = MXF8F6F4Format::E2M1 ✓
  - is_type_erased = false
            ↓
Kernel Signature:
  __global__ kernel(float_e2m1_t* A, ...)
                    ^^^^^^^^^^^^^^^
                    Known at compile time
            ↓
PTX Generation:
  tcgen05.mma.format::e2m1 ...  ← Hardcoded in PTX!
            ↓
Result:
  - Fastest (no runtime dispatch)
  - Smallest kernel (no format switching)
  - Largest binary (one kernel per format)
```

### Type-Erased Path (Runtime Format)

```
User Code:
  using ElementA = type_erased_dynamic_nv_float4_t;
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                   Union type
            ↓
Template Instantiation:
  CollectiveBuilder<type_erased_dynamic_float4_t, ...>
  - compile_time_format = MXF8F6F4Format::INVALID ✗
  - is_type_erased = true
            ↓
Kernel Signature:
  __global__ kernel(
      type_erased_dynamic_float4_t* A,
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      Union type (format unknown)
      MXF8F6F4Format runtime_format  ← Runtime argument!
  )
            ↓
PTX Generation:
  Format dispatch:
    switch (runtime_format) {
      case E2M1: tcgen05.mma.format::e2m1 ...
      case E2M3: tcgen05.mma.format::e2m3 ...
    }
            ↓
Result:
  - Slightly slower (minimal runtime dispatch)
  - Slightly larger kernel (format switching code)
  - Much smaller binary (single kernel for all formats)
```

## Performance Implications

### Overhead Analysis

**Static Type**:
```asm
; Direct PTX instruction (no dispatch)
tcgen05.mma.cta_group::2.format::e2m1 {accum}, {a}, {b}, {c};
; Latency: ~1 cycle (instruction issue)
```

**Type-Erased**:
```asm
; Runtime format dispatch
ld.param.u8  %r1, [format];     ; Load format (1 cycle)
setp.eq.u8   %p1, %r1, 5;       ; Compare to E2M1 (1 cycle)
@%p1 tcgen05.mma.format::e2m1 ...; ; Predicated MMA (1 cycle)
@!%p1 tcgen05.mma.format::e2m3 ...;
; Total overhead: ~2 cycles per MMA instruction
```

**Impact**:
- **Per-MMA overhead**: ~2 cycles
- **MMA latency**: ~200+ cycles
- **Relative overhead**: ~1% (negligible)

### Binary Size Reduction

**Example**: GEMM kernel supporting 3 formats

**Static Approach**:
```
Kernel for E2M1:  512 KB
Kernel for E2M3:  512 KB
Kernel for E3M2:  512 KB
────────────────────────
Total:           1536 KB
```

**Type-Erased Approach**:
```
Single kernel:    520 KB  (slightly larger due to dispatch code)
Savings:         1016 KB  (66% reduction!)
```

## Summary

Type-erased call chains enable:

1. **Single Kernel Binary**: One compiled kernel handles all formats
2. **Runtime Format Selection**: User chooses format at launch time
3. **Minimal Overhead**: ~1% performance cost for 66% binary size reduction
4. **Forward Compatibility**: Add new formats without recompiling

**Key Insight**: Type erasure moves format information from template parameters (compile-time) to function arguments (runtime), enabling dramatic binary size reduction with negligible performance cost.

---

**See Also**:
- [09-type-erased-float4.md](09-type-erased-float4.md) - Complete type erasure documentation
- [07-call-chains.md](07-call-chains.md) - Call chains for concrete (static) types
- [08-design-patterns.md](08-design-patterns.md) - Type erasure design pattern
