# CuTe DSL → MLIR → CuTe C++ Mapping Guide

This document provides a frame-by-frame trace of how CuTeDSL Python code maps to MLIR operations and their corresponding CuTe C++ types.

## Table of Contents

1. [Overview](#overview)
2. [HopperWgmmaGemmPersistentKernel Trace](#hopperwgmmagemmpersistentkernel-trace)
3. [Setup Attributes Deep Dive](#setup-attributes-deep-dive)
4. [Kernel Call Deep Dive](#kernel-call-deep-dive)
5. [MLIR Operations → CuTe C++ Mapping](#mlir-operations--cute-c-mapping)

---

## Overview

CuTe DSL provides a Python interface that directly generates PTX code through MLIR, using the same CuTe abstractions found in C++. The flow is:

```
Python CuTeDSL → MLIR IR → PTX Assembly
     ↓
CuTe C++ (conceptual mapping)
```

### Key Design Principles

1. **Direct MLIR Generation**: Python decorators (`@cute.jit`, `@cute.kernel`) trigger MLIR code generation
2. **Type Preservation**: Python types map directly to MLIR types which mirror CuTe C++ types
3. **Lazy Evaluation**: Operations build MLIR IR rather than executing immediately
4. **Protocol-Based Conversion**: Objects implement `__extract_mlir_values__()` for seamless conversion

---

## HopperWgmmaGemmPersistentKernel Trace

### File Location
[dense_gemm_persistent.py](../examples/python/CuTeDSL/hopper/dense_gemm_persistent.py)

### Class Initialization

```python
# Line 254-290
def __init__(
    self,
    acc_dtype: type[cutlass.Numeric],
    tile_shape_mn: tuple[int, int],
    cluster_shape_mn: tuple[int, int],
    swizzle_size: int,
    raster_along_m: bool,
):
```

**Frame-by-Frame Initialization:**

1. **Store configuration parameters** (lines 276-280)
   - `self.acc_dtype` - Accumulator data type (e.g., Float32)
   - `self.cluster_shape_mn` - Cluster dimensions for parallel processing
   - `self.tile_shape_mnk` - CTA tile shape (M, N, K) - K deferred to `_setup_attributes`

2. **Compute derived parameters** (lines 286-289)
   - `self.atom_layout_mnk` - Warp group layout for MMA atoms
   - Logic: Use 2 warp groups if tile is large (M>64 and N>128), else 1
   - Maps to CuTe C++ `Layout<Shape<_2,_1,_1>>` or `Layout<Shape<_1,_1,_1>>`

3. **Initialize memory and threading config** (lines 297-311)
   - `self.occupancy = 1` - CTAs per SM
   - `self.num_dma_warp_groups = 1` - DMA warps for loading
   - `self.num_mma_warp_groups` - MMA warps for computation
   - `self.threads_per_cta` - Total threads per CTA
   - `self.smem_capacity` - Shared memory capacity (~227KB on sm_90)

4. **Initialize stage and layout placeholders** (lines 313-329)
   - `self.ab_stage = None` - Pipeline stages for A/B matrices
   - `self.epi_stage = None` - Pipeline stages for epilogue
   - `self.a_smem_layout_staged = None` - Staged shared memory layouts
   - These are computed in `_setup_attributes()` after tensor properties are known

---

## Setup Attributes Deep Dive

### Entry Point: `_setup_attributes()`
[dense_gemm_persistent.py:331-406](../examples/python/CuTeDSL/hopper/dense_gemm_persistent.py#L331-L406)

This method configures attributes dependent on GEMM input properties. Called from `__call__` after tensor types are known.

### Frame 1: Make Tiled MMA

```python
# Line 351-359
self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
    self.a_dtype,
    self.b_dtype,
    self.a_layout.sm90_mma_major_mode(),
    self.b_layout.sm90_mma_major_mode(),
    self.acc_dtype,
    self.atom_layout_mnk,
    tiler_mn=(64, self.tile_shape_mnk[1]),
)
```

**Call Path:**
1. **Python**: `sm90_utils.make_trivial_tiled_mma()`
2. **Location**: [cutlass/utils/hopper_helpers.py](../python/CuTeDSL/cutlass/utils/hopper_helpers.py)
3. **Returns**: `cute.TiledMma` object

**Unpacking `make_trivial_tiled_mma`:**

```python
# hopper_helpers.py (conceptual - actual implementation varies)
def make_trivial_tiled_mma(
    a_dtype, b_dtype,
    a_major_mode, b_major_mode,
    acc_dtype, atom_layout_mnk, tiler_mn
):
    # Select MMA instruction based on data types
    mma_atom = select_mma_atom(a_dtype, b_dtype, acc_dtype)
    # e.g., for Float16 x Float16 -> Float32:
    # Returns SM90_64x128x16_F32F16F16_SS (WGMMA instruction)

    # Create tiled MMA by tiling the atom
    tiled_mma = cute.make_tiled_mma(
        mma_atom,
        atom_layout_mnk,  # e.g., (2,1,1) for 2 warp groups
        tiler_mn          # e.g., (64, 256)
    )
    return tiled_mma
```

**MLIR Generation:**
- **Type**: `!cute.tiled_mma<...>`
- **Operation**: `cute.make_tiled_mma`
- **Attributes**: MMA shape, operand layouts, accumulator type

**CuTe C++ Equivalent:**

```cpp
// include/cute/atom/mma_traits_sm90_gmma.hpp
using MMA_Atom = SM90_64x128x16_F32F16F16_SS<
  GMMA::Major::K, GMMA::Major::K>;

using TiledMMA = decltype(
  make_tiled_mma(
    MMA_Atom{},
    Layout<Shape<_2,_1,_1>>{},  // atom_layout_mnk
    Tile<_64, _256>{}           // tiler_mn
  )
);
```

**CuTe C++ File**:
- Atom: [include/cute/atom/mma_traits_sm90_gmma.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/mma_traits_sm90_gmma.hpp)
- Tiled MMA: [include/cute/algorithm/gemm.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/algorithm/gemm.hpp)

---

### Frame 2: Compute K Dimension

```python
# Line 360-366
mma_inst_shape_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])
mma_inst_tile_k = 4
self.tile_shape_mnk = (
    self.tile_shape_mnk[0],
    self.tile_shape_mnk[1],
    mma_inst_shape_k * mma_inst_tile_k,
)
```

**Call Path:**
1. **Python**: `cute.size(tensor, mode=[2])`
2. **Implementation**: [cutlass/cute/core.py](../python/CuTeDSL/cutlass/cute/core.py)

**Unpacking `cute.size()`:**

```python
# cutlass/cute/core.py
@dsl_user_op
def size(x, mode=None, *, loc=None, ip=None):
    """Get size of a shape/layout at given mode(s)"""
    if mode is None:
        # Return total size
        return _cute_ir.size(x, loc=loc, ip=ip)
    else:
        # Return size at specific mode
        return _cute_ir.size_with_mode(x, mode, loc=loc, ip=ip)
```

**MLIR Generation:**
- **Operation**: `cute.size` or `cute.size_with_mode`
- **Input**: `!cute.layout` or `!cute.shape`
- **Output**: `!cute.int_tuple<"?">` (constrained integer)

**CuTe C++ Equivalent:**

```cpp
// include/cute/numeric/int.hpp
template <class T>
CUTE_HOST_DEVICE constexpr auto
size(T const& t) {
  if constexpr (is_tuple<T>::value) {
    return product(t);  // Product of all elements
  } else {
    return t;
  }
}

// For specific mode:
template <class T, class... Modes>
CUTE_HOST_DEVICE constexpr auto
size(T const& t, Modes... modes) {
  return size(get<Modes...>(t));
}
```

**CuTe C++ File**: [include/cute/numeric/int.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/numeric/int.hpp)

---

### Frame 3: Compute CTA Layout

```python
# Line 368-372
self.cta_layout_mnk = cute.make_layout((*self.cluster_shape_mn, 1))
self.num_mcast_ctas_a = self.cluster_shape_mn[1]
self.num_mcast_ctas_b = self.cluster_shape_mn[0]
self.is_a_mcast = self.num_mcast_ctas_a > 1
self.is_b_mcast = self.num_mcast_ctas_b > 1
```

**Call Path:**
1. **Python**: `cute.make_layout((m, n, k))`
2. **Implementation**: [cutlass/cute/core.py](../python/CuTeDSL/cutlass/cute/core.py)

**Unpacking `cute.make_layout()`:**

```python
# cutlass/cute/core.py (simplified)
@dsl_user_op
def make_layout(shape, stride=None, *, loc=None, ip=None):
    """Create a layout from shape and optional stride"""
    # Pack shape into MLIR value
    shape_val = _pack_shape(shape, loc=loc, ip=ip)

    if stride is None:
        # Generate compact column-major stride
        layout_val = _cute_ir.make_layout(shape_val, loc=loc, ip=ip)
    else:
        # Use provided stride
        stride_val = _pack_stride(stride, loc=loc, ip=ip)
        layout_val = _cute_ir.make_layout_with_stride(
            shape_val, stride_val, loc=loc, ip=ip
        )

    return _Layout(layout_val)
```

**MLIR Generation:**

```mlir
// Input shape: (2, 1, 1)
%shape = cute.make_shape : () -> !cute.shape<"2", "1", "1">
%layout = cute.make_layout %shape : (!cute.shape<"2", "1", "1">)
          -> !cute.layout<(!cute.shape<"2", "1", "1">, !cute.stride<"1", "2", "2">)>
```

**CuTe C++ Equivalent:**

```cpp
// include/cute/layout.hpp
template <class Shape, class Stride>
struct Layout {
  Shape  shape_;
  Stride stride_;
};

// Factory function
template <class... Shapes>
CUTE_HOST_DEVICE constexpr auto
make_layout(Shape<Shapes...> const& shape) {
  // Generate compact column-major stride
  auto stride = compact_col_major(shape);
  return Layout<Shape<Shapes...>, decltype(stride)>{shape, stride};
}

// Example: make_layout(Shape<_2, _1, _1>{})
// Returns: Layout<Shape<_2,_1,_1>, Stride<_1,_2,_2>>
```

**CuTe C++ File**: [include/cute/layout.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/layout.hpp)

---

### Frame 4: Compute Epilogue Tile

```python
# Line 374-377
is_cooperative = self.atom_layout_mnk == (2, 1, 1)
self.epi_tile = self._sm90_compute_tile_shape_or_override(
    self.tile_shape_mnk, self.c_dtype, is_cooperative=is_cooperative
)
```

**Unpacking `_sm90_compute_tile_shape_or_override()`:**

```python
# Line 1028-1059
@staticmethod
def _sm90_compute_tile_shape_or_override(
    tile_shape_mnk: tuple[int, int, int],
    element_type: type[cutlass.Numeric],
    is_cooperative: bool = False,
    epi_tile_override: Optional[tuple[int, int]] = None,
) -> tuple[int, int]:
    if epi_tile_override is not None:
        return epi_tile_override

    if is_cooperative:
        # For 2 warp groups (cooperative)
        tile_m = min(128, cute.size(tile_shape_mnk, mode=[0]))
        tile_n = min(32, cute.size(tile_shape_mnk, mode=[1]))
        return (tile_m, tile_n)
    else:
        # For 1 warp group
        n_perf = 64 if element_type.width == 8 else 32
        tile_m = min(64, cute.size(tile_shape_mnk, mode=[0]))
        tile_n = min(n_perf, cute.size(tile_shape_mnk, mode=[1]))
        return (tile_m, tile_n)
```

**Explanation:**
- Epilogue tile is smaller than main CTA tile for efficient register→SMEM→GMEM pipeline
- Cooperative (2 warp groups): Use larger M tile (128) but smaller N tile (32)
- Non-cooperative (1 warp group): Use smaller M tile (64) and moderate N tile (32 or 64)
- For 8-bit types, use larger N tile (64) for better memory throughput

**CuTe C++ Equivalent:**
- This is a host-side heuristic, not a runtime computation
- In C++, you'd define this as a template parameter:

```cpp
template <int CtaTileM, int CtaTileN, bool IsCooperative>
struct EpilogueTileTraits {
  static constexpr int TileM = IsCooperative ? min(128, CtaTileM) : min(64, CtaTileM);
  static constexpr int TileN = IsCooperative ? min(32, CtaTileN) : min(32, CtaTileN);
  using Shape = Shape<Int<TileM>, Int<TileN>>;
};
```

---

### Frame 5: Compute Stage Counts

```python
# Line 380-388
self.ab_stage, self.epi_stage = self._compute_stages(
    self.tile_shape_mnk,
    self.a_dtype,
    self.b_dtype,
    self.epi_tile,
    self.c_dtype,
    self.smem_capacity,
    self.occupancy,
)
```

**Unpacking `_compute_stages()`:**

```python
# Line 980-1026
@staticmethod
def _compute_stages(
    tile_shape_mnk: tuple[int, int, int],
    a_dtype: type[cutlass.Numeric],
    b_dtype: type[cutlass.Numeric],
    epi_tile: tuple[int, int],
    c_dtype: type[cutlass.Numeric],
    smem_capacity: int,
    occupancy: int,
) -> tuple[int, int]:
    """Computes the number of pipeline stages for A/B/C operands"""

    # 1. Compute A/B bytes per stage
    a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))  # (M, K)
    b_shape = cute.slice_(tile_shape_mnk, (0, None, None))  # (N, K)
    ab_bytes_per_stage = (
        cute.size(a_shape) * a_dtype.width // 8
        + cute.size(b_shape) * b_dtype.width // 8
    )

    # 2. Compute epilogue bytes
    c_bytes_per_stage = cute.size(epi_tile) * c_dtype.width // 8
    epi_stage = 4  # Fixed at 4 stages for epilogue
    epi_bytes = c_bytes_per_stage * epi_stage

    # 3. Compute AB stages to fill remaining SMEM
    mbar_helpers_bytes = 1024  # Barrier storage overhead
    ab_stage = (
        smem_capacity // occupancy - (mbar_helpers_bytes + epi_bytes)
    ) // ab_bytes_per_stage

    return ab_stage, epi_stage
```

**Explanation:**

1. **`cute.slice_()` operation**:
   - Slices a shape/layout by replacing modes with integers or keeping with None
   - `cute.slice_((M, N, K), (None, 0, None))` → `(M, K)` (removes N dimension)
   - `cute.slice_((M, N, K), (0, None, None))` → `(N, K)` (removes M dimension)

2. **Stage computation**:
   - SMEM budget = Total SMEM / Occupancy
   - Reserve space for epilogue (4 stages) and barriers (1KB)
   - Fill remaining space with A/B pipeline stages
   - More stages = better latency hiding but more SMEM usage

**MLIR for `cute.slice_`:**

```mlir
%orig_shape = cute.make_shape : () -> !cute.shape<"128", "256", "64">
%slice_coord = cute.make_coord : () -> !cute.coord<"_", "0", "_">
%sliced_shape = cute.slice %orig_shape, %slice_coord
                : (!cute.shape<"128", "256", "64">, !cute.coord<"_", "0", "_">)
                -> !cute.shape<"128", "64">
```

**CuTe C++ Equivalent:**

```cpp
// include/cute/algorithm/tuple_algorithms.hpp
template <class Tuple, class Coord>
CUTE_HOST_DEVICE constexpr auto
slice(Tuple const& tuple, Coord const& coord) {
  return slice_impl(tuple, coord, make_seq<tuple_size<Tuple>::value>{});
}

// Example:
auto orig_shape = Shape<_128, _256, _64>{};
auto coord = make_coord(_, _0{}, _);  // _ means "keep this mode"
auto sliced = slice(orig_shape, coord);  // Returns Shape<_128, _64>
```

**CuTe C++ File**: [include/cute/algorithm/tuple_algorithms.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/algorithm/tuple_algorithms.hpp)

---

### Frame 6: Create Shared Memory Layouts

```python
# Line 390-405
(
    self.a_smem_layout_staged,
    self.b_smem_layout_staged,
    self.epi_smem_layout_staged,
) = self._make_smem_layouts(
    self.tile_shape_mnk,
    self.epi_tile,
    self.a_dtype,
    self.a_layout,
    self.b_dtype,
    self.b_layout,
    self.ab_stage,
    self.c_dtype,
    self.c_layout,
    self.epi_stage,
)
```

**Unpacking `_make_smem_layouts()` - Part 1: A Matrix Layout:**

```python
# Line 1062-1121
@staticmethod
def _make_smem_layouts(...) -> tuple[cute.ComposedLayout, ...]:
    # 1. Get A tile shape
    a_smem_shape = cute.slice_(tile_shape_mnk, (None, 0, None))  # (M, K)

    # 2. Determine if A is K-major or M-major
    a_is_k_major = (
        a_layout.sm90_mma_major_mode() == cute.nvgpu.warpgroup.OperandMajorMode.K
    )

    # 3. Get major mode size for swizzling
    a_major_mode_size = tile_shape_mnk[2 if a_is_k_major else 0]

    # 4. Create SMEM layout atom with swizzling
    a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(
            a_layout,
            a_dtype,
            a_major_mode_size,
        ),
        a_dtype,
    )

    # 5. Tile to full shape with pipeline stages
    a_smem_layout_staged = cute.tile_to_shape(
        a_smem_layout_atom,
        cute.append(a_smem_shape, ab_stage),  # Add stage dimension
        order=(0, 1, 2) if a_is_k_major else (1, 0, 2),
    )

    return a_smem_layout_staged, b_smem_layout_staged, epi_smem_layout_staged
```

**Deep Dive: `make_smem_layout_atom()`**

This is a **critical** operation that creates swizzled SMEM layouts for Hopper's WGMMA instructions.

```python
# cutlass/_mlir/dialects/cute_nvgpu.py (binding to C++)
def make_smem_layout_atom(swizzle, dtype):
    """Create SMEM layout atom with swizzle pattern for WGMMA"""
    # Returns ComposedLayout with inner swizzle function
    pass
```

**MLIR Generation:**

```mlir
%swizzle = cute.make_swizzle<128, 4, 3>  // SwizzleMode B=128, M=4, S=3
%atom_layout = cute.make_smem_layout_atom %swizzle, !type<f16>
               : () -> !cute.composed_layout<...>
```

**CuTe C++ Equivalent:**

```cpp
// include/cute/atom/copy_traits_sm90_tma.hpp
template <int ModeSize, class ElementType>
struct SmemLayoutAtom_SM90 {
  using SwizzleAtom = Swizzle<3, 4, 3>;  // B=8 bytes, M=4, S=3

  using Layout = decltype(
    composition(
      Swizzle<3, 4, 3>{},
      Layout<Shape<_128, _64>, Stride<_64, _1>>{}
    )
  );
};

// Swizzle transforms address to avoid bank conflicts:
// addr' = addr XOR (addr >> 4)
```

**Swizzle Explanation:**
- **B (Base)**: Number of bytes in swizzle base (8, 16, 32, 64, 128)
- **M (Mask)**: Controls XOR pattern for bank conflict avoidance
- **S (Shift)**: Bits to shift before XOR

For Hopper WGMMA with 128B swizzle:
```
Swizzle<3, 4, 3> means:
- Base = 2^3 = 8 bytes
- Mask = 4 bits (16 banks)
- Shift = 3 bits
Address transformation: addr' = addr ^ ((addr >> 3) & (4-1))
```

**CuTe C++ Files**:
- Layout atom: [include/cute/atom/copy_traits_sm90.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/copy_traits_sm90.hpp)
- Swizzle: [include/cute/swizzle.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/swizzle.hpp)

---

**Deep Dive: `cute.tile_to_shape()`**

```python
# cutlass/cute/core.py
@dsl_user_op
def tile_to_shape(layout, shape, order=None, *, loc=None, ip=None):
    """Tile a layout to match a target shape"""
    shape_val = _pack_shape(shape, loc=loc, ip=ip)

    if order is None:
        result = _cute_ir.tile_to_shape(layout, shape_val, loc=loc, ip=ip)
    else:
        order_val = _pack_int_tuple(order, loc=loc, ip=ip)
        result = _cute_ir.tile_to_shape_with_order(
            layout, shape_val, order_val, loc=loc, ip=ip
        )

    return _ComposedLayout(result)
```

**MLIR Generation:**

```mlir
// Tile atom layout to (M, K, Stages)
%atom = ... : !cute.composed_layout<...>
%target_shape = cute.make_shape : () -> !cute.shape<"128", "64", "7">
%order = cute.make_int_tuple : () -> !cute.int_tuple<"0", "1", "2">

%tiled = cute.tile_to_shape %atom, %target_shape, %order
         : (!cute.composed_layout<...>, !cute.shape<"128", "64", "7">, !cute.int_tuple<"0", "1", "2">)
         -> !cute.composed_layout<...>
```

**CuTe C++ Equivalent:**

```cpp
// include/cute/layout.hpp
template <class Layout, class Shape>
CUTE_HOST_DEVICE constexpr auto
tile_to_shape(Layout const& layout, Shape const& shape) {
  return tile_to_shape(layout, shape, GenColMajor{});
}

template <class Layout, class Shape, class Order>
CUTE_HOST_DEVICE constexpr auto
tile_to_shape(Layout const& layout, Shape const& shape, Order const& order) {
  // Repeat layout to cover target shape
  auto tiled_shape = shape_div(shape, layout.shape());
  return logical_divide(layout, tiled_shape)(_, order);
}

// Example:
auto atom = Layout<Shape<_32, _8>, Stride<_8, _1>, Swizzle<3,4,3>>{};
auto target = Shape<_128, _64, _7>{};  // Include 7 pipeline stages
auto tiled = tile_to_shape(atom, target, GenColMajor{});
// Result: Layout that repeats atom to cover (128, 64) with 7 stages
```

**CuTe C++ File**: [include/cute/layout.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/layout.hpp)

---

## Kernel Call Deep Dive

### Entry Point: `__call__()`
[dense_gemm_persistent.py:408-533](../examples/python/CuTeDSL/hopper/dense_gemm_persistent.py#L408-L533)

This method is the main entry point, decorated with `@cute.jit` to trigger JIT compilation.

```python
@cute.jit
def __call__(
    self,
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    max_active_clusters: cutlass.Constexpr,
    stream: cuda.CUstream,
):
```

### Frame 1: Extract Tensor Properties

```python
# Line 436-441
self.a_dtype = a.element_type
self.b_dtype = b.element_type
self.c_dtype = c.element_type
self.a_layout = utils.LayoutEnum.from_tensor(a)
self.b_layout = utils.LayoutEnum.from_tensor(b)
self.c_layout = utils.LayoutEnum.from_tensor(c)
```

**Unpacking `a.element_type`:**

From [cutlass/cute/runtime.py:106](../python/CuTeDSL/cutlass/cute/runtime.py#L106):

```python
class _Tensor(Tensor):
    @property
    def dtype(self) -> Type[Numeric]:
        return self._dtype
```

**Unpacking `utils.LayoutEnum.from_tensor()`:**

```python
# cutlass/utils/__init__.py (conceptual)
class LayoutEnum:
    @staticmethod
    def from_tensor(tensor):
        """Infer layout from tensor leading dimension"""
        leading_dim = tensor.leading_dim
        if tensor.shape[0] == leading_dim:
            return LayoutEnum.M_MAJOR  # Row-major
        elif tensor.shape[1] == leading_dim:
            return LayoutEnum.K_MAJOR or LayoutEnum.N_MAJOR
        # ... more logic
```

---

### Frame 2: Validate Data Types

```python
# Line 443-452
if cutlass.const_expr(
    self.a_dtype.width == 16 and self.a_dtype != self.b_dtype
):
    raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")
```

**Key Point**: `cutlass.const_expr()` evaluates expressions at compile time (during MLIR generation), not runtime. This enables:
- Static error checking
- Compile-time optimization
- Type specialization

---

### Frame 3: Setup Attributes

```python
# Line 454
self._setup_attributes()
```

This was fully traced in the [Setup Attributes Deep Dive](#setup-attributes-deep-dive) section.

---

### Frame 4: Create TMA Atoms and Tensors

```python
# Line 456-468
tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
    a,
    self.a_smem_layout_staged,
    (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
    self.cluster_shape_mn[1],
)
```

**Unpacking `_make_tma_atoms_and_tensors()`:**

```python
# Line 1226-1261
@staticmethod
def _make_tma_atoms_and_tensors(
    tensor: cute.Tensor,
    smem_layout_staged: cute.ComposedLayout,
    smem_tile: tuple[int, int],
    mcast_dim: int,
) -> tuple[cute.CopyAtom, cute.Tensor]:
    # 1. Choose TMA operation based on multicast dimension
    op = (
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        if mcast_dim == 1
        else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
    )

    # 2. Slice staged layout to get single-stage layout
    smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))

    # 3. Create tiled TMA atom
    tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
        op,
        tensor,
        smem_layout,
        smem_tile,
        num_multicast=mcast_dim,
    )

    return tma_atom, tma_tensor
```

**Deep Dive: `make_tiled_tma_atom()`**

This creates a TMA descriptor for async memory copy.

```python
# cutlass/_mlir/dialects/cute_nvgpu.py
def make_tiled_tma_atom(op, tensor, smem_layout, tile_shape, num_multicast):
    """Create TMA atom for tensor memory access"""
    # Generate TMA descriptor on host
    # Returns (CopyAtom, partitioned tensor)
    pass
```

**MLIR Generation:**

```mlir
%tma_desc = cute_nvgpu.make_tma_descriptor
            %tensor, %smem_layout, %tile_shape, %num_multicast
            : (!cute.tensor<...>, !cute.layout<...>, !cute.shape<...>, i32)
            -> !cute_nvgpu.tma_descriptor

%tma_atom = cute.make_copy_atom %tma_desc
            : (!cute_nvgpu.tma_descriptor) -> !cute.copy_atom<tma_g2s>
```

**CuTe C++ Equivalent:**

```cpp
// include/cute/arch/copy_sm90_tma.hpp
template <class GTensor, class SLayout, class TileShape>
auto make_tma_copy(GTensor const& gtensor,
                    SLayout const& slayout,
                    TileShape const& tile_shape) {
  // Create TMA descriptor
  using TMA_Op = Copy_Traits<SM90_TMA_LOAD>;

  // Partition tensor for TMA
  auto tma_atom = make_tiled_copy(
    Copy_Atom<TMA_Op>{},
    Layout<_1, _0>{},  // Thread layout (single thread issues TMA)
    tile_shape
  );

  return tma_atom;
}
```

**TMA Overview:**
- **Tensor Memory Accelerator** (TMA) is Hopper's hardware unit for async memory copy
- Directly transfers rectangular tiles: GMEM ↔ SMEM
- Supports:
  - Swizzling (for bank conflict avoidance)
  - Multicast (one CTA loads, multiple CTAs receive)
  - Asynchronous operation with barriers

**CuTe C++ Files**:
- TMA copy traits: [include/cute/arch/copy_sm90_tma.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm90_tma.hpp)
- TMA descriptor: [include/cute/arch/copy_sm90_desc.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm90_desc.hpp)

---

### Frame 5: Compute Grid Size

```python
# Line 476-483
tile_sched_params, grid = self._compute_grid(
    c,
    self.tile_shape_mnk,
    self.cluster_shape_mn,
    self.swizzle_size,
    self.raster_along_m,
    max_active_clusters,
)
```

**Unpacking `_compute_grid()`:**

```python
# Line 1159-1196
@staticmethod
def _compute_grid(
    c: cute.Tensor,
    tile_shape_mnk: tuple[int, int, int],
    cluster_shape_mn: tuple[int, int],
    swizzle_size: int,
    raster_along_m: bool,
    max_active_clusters: cutlass.Constexpr,
) -> tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]:
    # 1. Slice tile shape to get C tile
    c_shape = cute.slice_(tile_shape_mnk, (None, None, 0))  # (M, N)

    # 2. Divide C tensor by tile shape to get tile grid
    gc = cute.zipped_divide(c, tiler=c_shape)

    # 3. Extract number of CTAs per dimension
    num_ctas_mnl = gc[(0, (None, None, None))].shape
    cluster_shape_mnl = (*cluster_shape_mn, 1)

    # 4. Create tile scheduler parameters
    tile_sched_params = utils.PersistentTileSchedulerParams(
        num_ctas_mnl,
        cluster_shape_mnl,
        swizzle_size,
        raster_along_m,
    )

    # 5. Compute grid dimensions
    grid = utils.StaticPersistentTileScheduler.get_grid_shape(
        tile_sched_params, max_active_clusters
    )

    return tile_sched_params, grid
```

**Deep Dive: `cute.zipped_divide()`**

This is a **key** CuTe operation that divides a tensor into tiles.

```python
# cutlass/cute/core.py
@dsl_user_op
def zipped_divide(tensor, tiler, *, loc=None, ip=None):
    """Divide tensor into tiles"""
    tiler_val = _pack_tile(tiler, loc=loc, ip=ip)
    result = _cute_ir.zipped_divide(tensor.value, tiler_val, loc=loc, ip=ip)
    return _Tensor(result, dtype=tensor.element_type)
```

**MLIR Generation:**

```mlir
// Divide tensor C [M=8192, N=8192, L=1] by tile [128, 256]
%c_tensor = ... : !cute.tensor<[8192, 8192, 1], f16>
%tiler = cute.make_tile : () -> !cute.tile<"128", "256">
%tiled = cute.zipped_divide %c_tensor, %tiler
         : (!cute.tensor<...>, !cute.tile<...>)
         -> !cute.tensor<[(128, 64), (256, 32), 1], f16>
//                        ^^^^^^^^  ^^^^^^^^^^
//                        tile      rest
```

**CuTe C++ Equivalent:**

```cpp
// include/cute/tensor.hpp
template <class Tensor, class Tiler>
CUTE_HOST_DEVICE constexpr auto
zipped_divide(Tensor&& tensor, Tiler const& tiler) {
  // Divide tensor layout by tiler
  auto new_layout = zipped_divide(tensor.layout(), tiler);

  // Return new tensor with divided layout
  return make_tensor(tensor.data(), new_layout);
}

// Example:
Tensor c = make_tensor(ptr, Layout<Shape<_8192, _8192>>);
auto tiler = Shape<_128, _256>{};
auto tiled = zipped_divide(c, tiler);
// tiled.shape() == Shape<Shape<_128, _64>, Shape<_256, _32>>
//                        ^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^
//                        tile (128x64)    rest (256x32 tiles)
```

**Explanation:**
- `zipped_divide` creates a hierarchical layout: `(tile, rest_tiles)`
- For C [8192, 8192] ÷ tile [128, 256]:
  - Tile: [128, 256]
  - Rest: [64 tiles in M, 32 tiles in N]
  - Result shape: `((128, 64), (256, 32))`

**CuTe C++ File**: [include/cute/tensor.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/tensor.hpp)

---

### Frame 6: Define Shared Storage

```python
# Line 485-510
@cute.struct
class SharedStorage:
    mainloop_pipeline_array_ptr: cute.struct.MemRange[
        cutlass.Int64, self.ab_stage * 2
    ]
    sA: cute.struct.Align[
        cute.struct.MemRange[
            self.a_dtype, cute.cosize(self.a_smem_layout_staged)
        ],
        self.buffer_align_bytes,
    ]
    sB: cute.struct.Align[...]
    sC: cute.struct.Align[...]
```

**Unpacking `@cute.struct`:**

This decorator creates a structured type for shared memory allocation.

```python
# cutlass/_mlir/dialects/cute.py
def struct(cls):
    """Create MLIR struct type from Python class annotations"""
    # Parse class annotations
    # Generate MLIR struct type
    # Return wrapper class
    pass
```

**MLIR Generation:**

```mlir
!shared_storage = !llvm.struct<(
  // Pipeline barriers
  !llvm.array<14 x i64>,  // 7 AB stages * 2 barriers

  // A buffer: 128*64*7 elements * 2 bytes (fp16) = 114688 bytes
  !llvm.array<57344 x f16>,

  // B buffer: 256*64*7 elements * 2 bytes = 229376 bytes
  !llvm.array<114688 x f16>,

  // C epilogue buffer: 128*32*4 stages * 2 bytes = 32768 bytes
  !llvm.array<16384 x f16>
)>
```

**CuTe C++ Equivalent:**

```cpp
// C++ shared memory layout
template <class AType, class BType, class CType,
          int AStages, int BStages, int CStages>
struct SharedStorage {
  // Pipeline barriers
  uint64_t pipeline_barriers[AStages * 2];

  // A buffer: aligned to 1024 bytes
  alignas(1024) AType sA[128][64][AStages];

  // B buffer: aligned to 1024 bytes
  alignas(1024) BType sB[256][64][BStages];

  // C epilogue buffer: aligned to 1024 bytes
  alignas(1024) CType sC[128][32][CStages];
};
```

**Key Points:**
- `cute.struct.MemRange[dtype, count]` → Array allocation
- `cute.struct.Align[..., bytes]` → Alignment constraint
- `cute.cosize()` → Total elements in layout (opposite of size)

---

### Frame 7: Launch Kernel

```python
# Line 513-532
self.kernel(
    tma_atom_a, tma_tensor_a,
    tma_atom_b, tma_tensor_b,
    tma_atom_c, tma_tensor_c,
    self.tiled_mma,
    self.cta_layout_mnk,
    self.a_smem_layout_staged,
    self.b_smem_layout_staged,
    self.epi_smem_layout_staged,
    tile_sched_params,
).launch(
    grid=grid,
    block=[self.threads_per_cta, 1, 1],
    cluster=(*self.cluster_shape_mn, 1),
    min_blocks_per_mp=1,
    stream=stream,
)
```

**Call Path:**
1. `self.kernel(...)` - Calls device kernel (decorated with `@cute.kernel`)
2. `.launch(...)` - Configures and launches CUDA kernel

**MLIR Generation (High-Level):**

```mlir
// 1. Define device function
gpu.func @kernel(
  %tma_a: !cute_nvgpu.tma_descriptor,
  %tensor_a: !cute.tensor<...>,
  ...
) kernel {
  // Kernel body (traced in next section)
}

// 2. Launch kernel
gpu.launch_func @kernel
  blocks(%grid_x, %grid_y, %grid_z)
  threads(%threads_x, %threads_y, %threads_z)
  cluster(%cluster_x, %cluster_y, %cluster_z)
  args(%tma_a, %tensor_a, ...)
```

---

## Kernel Device Code Trace

### Entry Point: `kernel()`
[dense_gemm_persistent.py:536-978](../examples/python/CuTeDSL/hopper/dense_gemm_persistent.py#L536-L978)

```python
@cute.kernel
def kernel(
    self,
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nkl: cute.Tensor,
    tma_atom_c: cute.CopyAtom,
    mC_mnl: cute.Tensor,
    tiled_mma: cute.TiledMma,
    cta_layout_mnk: cute.Layout,
    a_smem_layout_staged: cute.ComposedLayout,
    b_smem_layout_staged: cute.ComposedLayout,
    epi_smem_layout_staged: cute.ComposedLayout,
    tile_sched_params: utils.PersistentTileSchedulerParams,
):
```

### Kernel Frame 1: Get Thread/Warp/CTA IDs

```python
# Line 581-583
tidx, _, _ = cute.arch.thread_idx()
warp_idx = cute.arch.warp_idx()
warp_idx = cute.arch.make_warp_uniform(warp_idx)
```

**MLIR Operations:**

```mlir
%tidx = gpu.thread_id x : index
%warp_idx_raw = arith.divui %tidx, c32 : index
%warp_idx = cute.make_warp_uniform %warp_idx_raw : index
```

**CuTe C++ Equivalent:**

```cpp
// include/cute/arch/util.hpp
CUTE_DEVICE int thread_idx() {
  return threadIdx.x + threadIdx.y * blockDim.x
         + threadIdx.z * blockDim.x * blockDim.y;
}

CUTE_DEVICE int warp_idx() {
  return thread_idx() / 32;
}

// Warp-uniform: Broadcast value from lane 0 to all lanes
template <class T>
CUTE_DEVICE T make_warp_uniform(T value) {
  return __shfl_sync(0xffffffff, value, 0);
}
```

**CuTe C++ File**: [include/cute/arch/util.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/util.hpp)

---

### Kernel Frame 2: Prefetch TMA Descriptors

```python
# Line 586-589
if warp_idx == 0:
    cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
    cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
    cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)
```

**MLIR Operations:**

```mlir
%is_warp0 = arith.cmpi eq, %warp_idx, c0 : index
scf.if %is_warp0 {
  cute_nvgpu.prefetch_tma_descriptor %tma_a
  cute_nvgpu.prefetch_tma_descriptor %tma_b
  cute_nvgpu.prefetch_tma_descriptor %tma_c
}
```

**CuTe C++ Equivalent:**

```cpp
// include/cute/arch/copy_sm90_tma.hpp
template <class TMA_Desc>
CUTE_DEVICE void prefetch_tma_descriptor(TMA_Desc const& desc) {
  if (threadIdx.x == 0) {
    uint64_t desc_addr = reinterpret_cast<uint64_t>(&desc);
    asm volatile("prefetch.tensormap [%0];" :: "l"(desc_addr));
  }
}
```

**Explanation:**
- Prefetching TMA descriptors improves latency of first TMA load
- Only one thread per block needs to prefetch
- PTX instruction: `prefetch.tensormap`

**CuTe C++ File**: [include/cute/arch/copy_sm90_tma.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm90_tma.hpp)

---

### Kernel Frame 3: Compute Cluster Coordinates

```python
# Line 591-594
cta_rank_in_cluster = cute.arch.make_warp_uniform(
    cute.arch.block_idx_in_cluster()
)
cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)
```

**Unpacking `block_idx_in_cluster()`:**

```python
# cutlass/_mlir/dialects/cute_nvgpu.py
def block_idx_in_cluster():
    """Get CTA's rank within cluster (0 to cluster_size-1)"""
    pass
```

**MLIR Operations:**

```mlir
%cta_rank_raw = cute_nvgpu.block_idx_in_cluster : () -> index
%cta_rank = cute.make_warp_uniform %cta_rank_raw : index

%cluster_coord = cute.get_flat_coord %cta_layout, %cta_rank
                 : (!cute.layout<...>, index) -> !cute.coord<...>
```

**CuTe C++ Equivalent:**

```cpp
// include/cute/arch/cluster_sm90.hpp
CUTE_DEVICE uint32_t block_idx_in_cluster() {
  uint32_t rank;
  asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(rank));
  uint32_t rank_y, rank_z;
  asm volatile("mov.u32 %0, %%cluster_ctaid.y;" : "=r"(rank_y));
  asm volatile("mov.u32 %0, %%cluster_ctaid.z;" : "=r"(rank_z));

  uint32_t dim_x, dim_y;
  asm volatile("mov.u32 %0, %%cluster_nctaid.x;" : "=r"(dim_x));
  asm volatile("mov.u32 %0, %%cluster_nctaid.y;" : "=r"(dim_y));

  return rank + rank_y * dim_x + rank_z * dim_x * dim_y;
}

// get_flat_coord converts rank to multi-dimensional coordinate
template <class Layout>
CUTE_DEVICE auto get_flat_coord(Layout const& layout, int rank) {
  return layout.get_hier_coord(rank);
}
```

**Explanation:**
- Hopper introduces **Thread Block Clusters**: Groups of CTAs that can share data
- `block_idx_in_cluster()` returns CTA's position within cluster (0-3 for 2×2 cluster)
- `get_flat_coord()` converts linear rank → (m, n, k) coordinate

**CuTe C++ File**: [include/cute/arch/cluster_sm90.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/cluster_sm90.hpp)

---

### Kernel Frame 4: Compute Multicast Masks

```python
# Line 596-604
a_mcast_mask = cute.make_layout_image_mask(
    cta_layout_mnk, cluster_coord_mnk, mode=1
)
b_mcast_mask = cute.make_layout_image_mask(
    cta_layout_mnk, cluster_coord_mnk, mode=0
)

a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0
```

**Unpacking `make_layout_image_mask()`:**

```python
# cutlass/cute/core.py
@dsl_user_op
def make_layout_image_mask(layout, coord, mode, *, loc=None, ip=None):
    """Create bitmask for multicast TMA"""
    coord_val = _pack_coord(coord, loc=loc, ip=ip)
    mask = _cute_ir.make_layout_image_mask(
        layout.value, coord_val, mode, loc=loc, ip=ip
    )
    return mask
```

**MLIR Operations:**

```mlir
// For 2x1 cluster (2 CTAs in M, 1 CTA in N):
// CTA (0,0,0) wants to multicast A to CTAs sharing same N column
%layout = ... : !cute.layout<Shape<_2,_1,_1>, Stride<_1,_2,_2>>
%coord = cute.make_coord : () -> !cute.coord<"0", "0", "0">

%a_mask = cute.make_layout_image_mask %layout, %coord, mode=1
          : (!cute.layout<...>, !cute.coord<...>, i32) -> i16
// Result: 0b11 (both CTAs in column receive A)

%b_mask = cute.make_layout_image_mask %layout, %coord, mode=0
          : (!cute.layout<...>, !cute.coord<...>, i32) -> i16
// Result: 0b01 (only CTA 0 receives B, no multicast needed)
```

**CuTe C++ Equivalent:**

```cpp
// include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp
template <class LayoutMNK, class CoordMNK, int Mode>
CUTE_DEVICE uint16_t make_layout_image_mask(
    LayoutMNK const& layout,
    CoordMNK const& coord,
    Int<Mode> mode) {
  // Get CTAs sharing same coordinate in Mode dimension
  uint16_t mask = 0;

  // For 2x1 cluster, mode=1 (N):
  // CTA (0,0) → mask = 0b11 (CTAs 0,1 share N=0)
  // CTA (1,0) → mask = 0b11 (CTAs 0,1 share N=0)

  for (int cta = 0; cta < size(layout); ++cta) {
    auto cta_coord = layout.get_hier_coord(cta);
    if (get<Mode>(cta_coord) == get<Mode>(coord)) {
      mask |= (1 << cta);
    }
  }

  return mask;
}
```

**Explanation:**
- **TMA Multicast**: One CTA loads data, hardware broadcasts to multiple CTAs
- **A multicast (mode=1)**: CTAs in same N column receive same A tile
- **B multicast (mode=0)**: CTAs in same M row receive same B tile
- Reduces L2 cache traffic by ~cluster_size

---

### Kernel Frame 5: Allocate Shared Memory

```python
# Line 612-613
smem = cutlass.utils.SmemAllocator()
storage = smem.allocate(self.shared_storage)
```

**Unpacking `SmemAllocator`:**

```python
# cutlass/utils/__init__.py
class SmemAllocator:
    def allocate(self, struct_type):
        """Allocate shared memory for struct type"""
        # Returns Python object with same structure
        # Each field is a view into shared memory
        pass
```

**MLIR Operations:**

```mlir
// Allocate shared memory buffer
%smem_size = cute.cosize !shared_storage : i32
%smem_base = llvm.mlir.addressof @__shared_storage
             : !llvm.ptr<3>  // address space 3 = shared memory

// Cast to struct type
%storage = llvm.bitcast %smem_base
           : !llvm.ptr<3> to !llvm.ptr<!shared_storage, 3>
```

**CuTe C++ Equivalent:**

```cpp
// Shared memory allocation in C++
__shared__ SharedStorage<...> shared_storage;

// Or dynamically:
extern __shared__ uint8_t smem[];
auto* storage = reinterpret_cast<SharedStorage<...>*>(smem);
```

---

### Kernel Frame 6: Initialize Pipeline

```python
# Line 616-639
mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
    pipeline.Agent.Thread
)

mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
consumer_arrive_cnt = (
    mcast_size * self.num_mma_warp_groups * self.num_warps_per_warp_group
)
mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
    pipeline.Agent.Thread, consumer_arrive_cnt
)

mainloop_pipeline = pipeline.PipelineTmaAsync.create(
    barrier_storage=mainloop_pipeline_array_ptr,
    num_stages=self.ab_stage,
    producer_group=mainloop_pipeline_producer_group,
    consumer_group=mainloop_pipeline_consumer_group,
    tx_count=tma_copy_bytes,
    cta_layout_vmnk=cute.make_layout((1, *cta_layout_mnk.shape)),
    defer_sync=True,
)
```

**Unpacking `PipelineTmaAsync.create()`:**

This creates an **asynchronous pipeline** for producer-consumer synchronization using Hopper's `mbarrier` (barrier with transaction count).

```python
# cutlass/pipeline/sm90.py (simplified)
class PipelineTmaAsync:
    @staticmethod
    def create(barrier_storage, num_stages, producer_group,
               consumer_group, tx_count, cta_layout_vmnk, defer_sync):
        # Initialize barriers
        for stage in range(num_stages):
            barrier_init(
                barrier_storage[stage * 2],      # Full barrier
                producer_group.thread_count,
                tx_count
            )
            barrier_init(
                barrier_storage[stage * 2 + 1],  # Empty barrier
                consumer_group.thread_count,
                tx_count
            )

        return PipelineTmaAsync(...)
```

**MLIR Operations:**

```mlir
// Initialize mbarrier for stage 0
%barrier_ptr = llvm.getelementptr %storage[0, 0]
               : (!llvm.ptr<!shared_storage, 3>)
               -> !llvm.ptr<i64, 3>

cute_nvgpu.mbarrier_init %barrier_ptr, %thread_count, %tx_count
  : (!llvm.ptr<i64, 3>, i32, i32) -> ()
```

**CuTe C++ Equivalent:**

```cpp
// include/cute/arch/copy_sm90_tma.hpp
template <int Stages>
struct PipelineTmaAsync {
  struct alignas(8) MBarrier {
    uint64_t barrier;
  };

  MBarrier full_barriers[Stages];
  MBarrier empty_barriers[Stages];

  // Initialize barriers
  CUTE_DEVICE void init(int thread_count, int tx_count) {
    for (int stage = 0; stage < Stages; ++stage) {
      if (threadIdx.x == 0) {
        uint64_t* full_ptr = &full_barriers[stage].barrier;
        asm volatile("mbarrier.init.shared.b64 [%0], %1;"
                     :: "r"(full_ptr), "r"(thread_count));

        uint64_t* empty_ptr = &empty_barriers[stage].barrier;
        asm volatile("mbarrier.init.shared.b64 [%0], %1;"
                     :: "r"(empty_ptr), "r"(thread_count));
      }
    }
  }
};
```

**Pipeline Explanation:**

1. **Two-Phase Barrier**:
   - **Full barrier**: Tracks when stage is full (data loaded)
   - **Empty barrier**: Tracks when stage is empty (data consumed)

2. **Transaction Count** (`tx_count`):
   - Number of bytes to be transferred
   - TMA automatically decrements barrier when transfer completes
   - When count reaches 0, barrier arrives

3. **Producer-Consumer Flow**:
   ```
   Producer (DMA warp):
     1. Wait on empty barrier (stage available)
     2. Issue TMA load → auto-arrives on full barrier
     3. Move to next stage

   Consumer (MMA warps):
     1. Wait on full barrier (data ready)
     2. Consume data (WGMMA)
     3. Arrive on empty barrier
   ```

**CuTe C++ File**: [include/cute/arch/copy_sm90_tma.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm90_tma.hpp)

---

### Kernel Frame 7: Cluster Synchronization

```python
# Line 642
pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)
```

**Unpacking `pipeline_init_arrive()`:**

```python
# cutlass/pipeline/__init__.py
def pipeline_init_arrive(cluster_shape_mn, is_relaxed):
    """All CTAs in cluster arrive at initialization barrier"""
    cute.arch.cluster_arrive(is_relaxed)
```

**MLIR Operations:**

```mlir
cute_nvgpu.cluster_arrive_relaxed : () -> ()
// Or: cute_nvgpu.cluster_arrive : () -> ()
```

**CuTe C++ Equivalent:**

```cpp
// include/cute/arch/cluster_sm90.hpp
CUTE_DEVICE void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;");
}

CUTE_DEVICE void cluster_arrive() {
  asm volatile("barrier.cluster.arrive.aligned;");
}
```

**Explanation:**
- **Cluster barrier**: Synchronizes all CTAs in cluster
- **Relaxed**: Doesn't wait, just signals arrival
- Used after barrier initialization to ensure all CTAs ready before starting

**Later in code:**

```python
# Line 722
pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
```

```cpp
CUTE_DEVICE void cluster_wait() {
  asm volatile("barrier.cluster.wait.aligned;");
}
```

---

### Kernel Frame 8: Create SMEM Tensors

```python
# Line 645-653
sA = storage.sA.get_tensor(
    a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
)
sB = storage.sB.get_tensor(
    b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
)
sC = storage.sC.get_tensor(
    epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
)
```

**Unpacking `get_tensor()`:**

```python
# cutlass/utils/smem_allocator.py (conceptual)
class MemRange:
    def get_tensor(self, layout, swizzle=None):
        """Create tensor view over shared memory buffer"""
        ptr = cute.make_ptr(
            self.dtype,
            self.address,
            cute.AddressSpace.smem
        )

        if swizzle is not None:
            composed_layout = cute.composition(swizzle, layout)
            return cute.make_tensor(ptr, composed_layout)
        else:
            return cute.make_tensor(ptr, layout)
```

**MLIR Operations:**

```mlir
// Get pointer to sA buffer
%sA_ptr = llvm.getelementptr %storage[0, 1]  // Field index 1 = sA
          : (!llvm.ptr<!shared_storage, 3>) -> !llvm.ptr<f16, 3>

// Create pointer object
%sA_cute_ptr = cute.make_ptr %sA_ptr, %dtype, %addr_space
               : (!llvm.ptr<f16, 3>, !type, !addr_space)
               -> !cute.ptr<f16, smem, 1024>

// Create composed layout (swizzle ∘ outer_layout)
%sA_layout = cute.composition %swizzle, %outer_layout
             : (!cute.swizzle<...>, !cute.layout<...>)
             -> !cute.composed_layout<...>

// Create tensor
%sA = cute.make_view %sA_cute_ptr, %sA_layout
      : (!cute.ptr<...>, !cute.composed_layout<...>)
      -> !cute.tensor<[(128,64,7)], f16, smem>
```

**CuTe C++ Equivalent:**

```cpp
// Get pointer to sA in shared memory
auto sA_ptr = cute::make_smem_ptr(&shared_storage.sA[0][0][0]);

// Create outer layout
auto outer_layout = Layout<Shape<_128,_64,_7>, Stride<...>>{};

// Create swizzle function
auto swizzle = Swizzle<3, 4, 3>{};

// Compose: swizzle ∘ outer_layout
auto composed_layout = composition(swizzle, outer_layout);

// Create tensor
auto sA = make_tensor(sA_ptr, composed_layout);
// Type: Tensor<float16*, ComposedLayout<Swizzle<...>, Layout<...>>>
```

**CuTe C++ Files**:
- Composition: [include/cute/layout.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/layout.hpp)
- Make tensor: [include/cute/tensor.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/tensor.hpp)

---

### Kernel Frame 9: Partition Global Tensors with `local_tile`

```python
# Line 657-673
gA_mkl = cute.local_tile(
    mA_mkl,
    cute.slice_(self.tile_shape_mnk, (None, 0, None)),
    (None, None, None),
)
gB_nkl = cute.local_tile(
    mB_nkl,
    cute.slice_(self.tile_shape_mnk, (0, None, None)),
    (None, None, None),
)
gC_mnl = cute.local_tile(
    mC_mnl,
    cute.slice_(self.tile_shape_mnk, (None, None, 0)),
    (None, None, None),
)
```

**Unpacking `cute.local_tile()`:**

```python
# cutlass/cute/core.py
@dsl_user_op
def local_tile(tensor, tile_shape, tile_coord, *, loc=None, ip=None):
    """Partition tensor into tiles and select tile at coordinate"""
    tile_shape_val = _pack_shape(tile_shape, loc=loc, ip=ip)
    tile_coord_val = _pack_coord(tile_coord, loc=loc, ip=ip)

    result = _cute_ir.local_tile(
        tensor.value, tile_shape_val, tile_coord_val, loc=loc, ip=ip
    )

    return _Tensor(result, dtype=tensor.element_type)
```

**MLIR Operations:**

```mlir
// Partition A: [M=8192, K=8192, L=1] into tiles of [128, 64]
%mA = ... : !cute.tensor<[8192, 8192, 1], f16>
%tile_shape = cute.make_shape : () -> !cute.shape<"128", "_", "64">
%tile_coord = cute.make_coord : () -> !cute.coord<"_", "_", "_">

%gA = cute.local_tile %mA, %tile_shape, %tile_coord
      : (!cute.tensor<...>, !cute.shape<...>, !cute.coord<...>)
      -> !cute.tensor<[(128, 64), 64, (64, 128), 64, 1], f16>
//                     ^^^^^^^^^^                  ^^^^^^^^^^
//                     tile shape                  rest tiles
```

**CuTe C++ Equivalent:**

```cpp
// include/cute/tensor.hpp
template <class Tensor, class TileShape, class TileCoord>
CUTE_HOST_DEVICE constexpr auto
local_tile(Tensor&& tensor, TileShape const& tile_shape,
           TileCoord const& tile_coord) {
  // Create hierarchical layout: (tile, rest_tiles_m, rest_tiles_k, rest_l)
  auto tiled_layout = logical_divide(tensor.layout(), tile_shape);

  // Select tile at coordinate (all modes in this case)
  return tensor(tile_coord);
}

// Example usage:
Tensor mA = make_tensor(ptr, Layout<Shape<_8192, _8192, _1>>);
auto tile_shape = Shape<_128, _, _64>{};  // _ means "keep full dimension"
auto coord = make_coord(_, _, _);  // Select all tiles

auto gA = local_tile(mA, tile_shape, coord);
// Result shape: Shape<Shape<_128, _64>, _64, Shape<_64, _128>, _64, _1>
//                     ^^^^^^^^^^^^^^  ^^^  ^^^^^^^^^^^^^^^^  ^^^  ^^^
//                     tile            K    rest tiles        K    L
```

**Explanation:**

`local_tile` creates a **5-mode** tensor from 3-mode input:
1. **Mode 0**: Tile shape in M (128)
2. **Mode 1**: K blocks within tile (64)
3. **Mode 2**: Remaining tiles in M (64 tiles)
4. **Mode 3**: Remaining K blocks (128 blocks)
5. **Mode 4**: Batch dimension (1)

This allows easy iteration:
```python
for batch in range(L):
    for m_tile in range(rest_m):
        for k_block in range(rest_k):
            tile = gA_mkl[(None, None, m_tile, k_block, batch)]
            # tile has shape (128, 64)
```

**CuTe C++ File**: [include/cute/tensor.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/tensor.hpp)

---

### Kernel Frame 10: TMA Partition

```python
# Line 679-695
a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
a_cta_crd = cluster_coord_mnk[1]
tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
    tma_atom_a,
    a_cta_crd,
    a_cta_layout,
    cute.group_modes(sA, 0, 2),
    cute.group_modes(gA_mkl, 0, 2),
)
```

**Unpacking `tma_partition()`:**

This is a **crucial** operation that partitions tensors for TMA copy.

```python
# cutlass/_mlir/dialects/cute_nvgpu.py
def tma_partition(tma_atom, cta_coord, cta_layout,
                  smem_tensor, gmem_tensor):
    """Partition SMEM and GMEM tensors for TMA copy"""
    # Returns: (smem_partition, gmem_partition)
    pass
```

**Unpacking `cute.group_modes()`:**

```python
# cutlass/cute/core.py
@dsl_user_op
def group_modes(tensor, begin_mode, end_mode, *, loc=None, ip=None):
    """Group modes [begin, end) into a single mode"""
    result = _cute_ir.group_modes(
        tensor.value, begin_mode, end_mode, loc=loc, ip=ip
    )
    return _Tensor(result, dtype=tensor.element_type)
```

**MLIR Operations:**

```mlir
// Group sA modes 0-2: (128, 64, 7) → ((128, 64), 7)
%sA = ... : !cute.tensor<[128, 64, 7], f16, smem>
%sA_grouped = cute.group_modes %sA, 0, 2
              : (!cute.tensor<[128, 64, 7], ...>)
              -> !cute.tensor<[(128, 64), 7], ...>

// TMA partition
%tAsA, %tAgA = cute_nvgpu.tma_partition
               %tma_atom, %cta_crd, %cta_layout, %sA_grouped, %gA_grouped
               : (...) -> (!cute.tensor<...>, !cute.tensor<...>)
```

**CuTe C++ Equivalent:**

```cpp
// include/cute/algorithm/copy.hpp
template <class TMA_Atom, class CTACoord, class CTALayout,
          class STensor, class GTensor>
CUTE_DEVICE auto
tma_partition(TMA_Atom const& tma,
              CTACoord const& cta_coord,
              CTALayout const& cta_layout,
              STensor&& s_tensor,
              GTensor&& g_tensor) {
  // Partition SMEM tensor per CTA
  auto tAsA = local_partition(tma, s_tensor, cta_coord, cta_layout);

  // Partition GMEM tensor per CTA
  auto tAgA = local_partition(tma, g_tensor, cta_coord, cta_layout);

  return cute::make_tuple(tAsA, tAgA);
}

// group_modes: Flatten modes [begin, end) into single mode
template <class Tensor, int Begin, int End>
CUTE_DEVICE auto
group_modes(Tensor&& tensor, Int<Begin>, Int<End>) {
  auto layout = tensor.layout();
  auto new_layout = group<Begin, End>(layout);
  return make_tensor(tensor.data(), new_layout);
}

// Example:
Tensor sA = ... // Shape<_128, _64, _7>
auto grouped = group_modes(sA, Int<0>{}, Int<2>{});
// Result: Shape<Shape<_128, _64>, _7>
```

**Explanation:**

1. **`group_modes(sA, 0, 2)`**:
   - Input: `Shape<128, 64, 7>` (M, K, Stage)
   - Output: `Shape<(128, 64), 7>` (Tile, Stage)
   - Groups spatial modes (M, K) for TMA transfer

2. **`tma_partition()`**:
   - Partitions tensors based on CTA's position in cluster
   - For 2×1 cluster:
     - CTA 0 gets tiles [0, 2, 4, ...] in N
     - CTA 1 gets tiles [1, 3, 5, ...] in N
   - Returns source and destination partitions for TMA

**CuTe C++ Files**:
- TMA partition: [include/cute/algorithm/copy.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/algorithm/copy.hpp)
- Group modes: [include/cute/algorithm/tuple_algorithms.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/algorithm/tuple_algorithms.hpp)

---

### Kernel Frame 11: Partition for MMA

```python
# Line 699-717
warp_group_idx = cute.arch.make_warp_uniform(
    tidx // self.num_threads_per_warp_group
)
mma_warp_group_thread_layout = cute.make_layout(
    self.num_mma_warp_groups, stride=self.num_threads_per_warp_group
)
thr_mma = tiled_mma.get_slice(
    mma_warp_group_thread_layout(warp_group_idx - self.num_dma_warp_groups)
)

tCsA = thr_mma.partition_A(sA)
tCsB = thr_mma.partition_B(sB)
tCrA = tiled_mma.make_fragment_A(tCsA)
tCrB = tiled_mma.make_fragment_B(tCsB)

tCgC = thr_mma.partition_C(gC_mnl)
acc_shape = tCgC.shape[:3]
accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
```

**Unpacking `tiled_mma.get_slice()`:**

```python
# cutlass/_mlir/dialects/cute.py
class TiledMma:
    def get_slice(self, thread_idx):
        """Get per-thread partition of TiledMMA"""
        return ThreadMma(...)
```

**Unpacking `partition_A()` and `make_fragment_A()`:**

These operations partition shared memory and create register fragments for WGMMA.

```python
# cutlass/_mlir/dialects/cute.py
class ThreadMma:
    def partition_A(self, tensor):
        """Partition A tensor for this thread"""
        pass

    def partition_B(self, tensor):
        """Partition B tensor for this thread"""
        pass

    def partition_C(self, tensor):
        """Partition C tensor for this thread"""
        pass

class TiledMma:
    def make_fragment_A(self, partition):
        """Create register fragment for A"""
        pass

    def make_fragment_B(self, partition):
        """Create register fragment for B"""
        pass
```

**MLIR Operations:**

```mlir
// Get thread's slice of TiledMMA
%warp_group_idx = arith.divui %tidx, c128 : index  // 128 threads per warp group
%thr_idx = ... : index  // Thread index within MMA warp groups

%thr_mma = cute.get_thr_mma %tiled_mma, %thr_idx
           : (!cute.tiled_mma<...>, index) -> !cute.thr_mma<...>

// Partition sA for this thread
%tCsA = cute.partition_A %thr_mma, %sA
        : (!cute.thr_mma<...>, !cute.tensor<...>)
        -> !cute.tensor<[..., stage], f16, smem>

// Create register fragment
%tCrA = cute.make_fragment_A %tiled_mma, %tCsA
        : (!cute.tiled_mma<...>, !cute.tensor<...>)
        -> !cute.tensor<[...], f16, rmem>
```

**CuTe C++ Equivalent:**

```cpp
// include/cute/atom/mma_traits_sm90_gmma.hpp
template <class TiledMMA>
struct MMA_Traits {
  // Get thread's partition
  CUTE_DEVICE auto get_slice(int thread_idx) const {
    return ThrMMA<...>{thread_idx};
  }
};

template <class ThrMMA, class Tensor>
CUTE_DEVICE auto partition_A(ThrMMA const& thr_mma, Tensor&& tensor) {
  // Repartition tensor based on MMA atom's layout
  return local_partition(tensor, thr_mma.layout_A(), thr_mma.coord());
}

template <class TiledMMA, class Partition>
CUTE_DEVICE auto make_fragment_A(TiledMMA const& mma, Partition const& part) {
  // Create register tensor with correct layout for WGMMA
  auto rmem_layout = mma.layout_A_fragment();
  return make_tensor<rmem>(rmem_layout, part.dtype());
}

// Example:
TiledMMA<SM90_64x128x16_F32F16F16_SS, Layout<Shape<_2,_1,_1>>> mma;

// Each warp group gets a slice
auto thr_mma = mma.get_slice(thread_idx);

// Partition sA (SMEM) for this thread
auto tCsA = thr_mma.partition_A(sA);  // Shape<...>

// Create register fragment
auto tCrA = mma.make_fragment_A(tCsA);  // Shape<...> in registers
```

**Explanation:**

Hopper WGMMA operates on **entire warp groups** (128 threads), not individual threads:

1. **`get_slice(thread_idx)`**:
   - Returns per-thread view of MMA operation
   - Thread layout typically: `Layout<Shape<_128>, Stride<_1>>` (all 128 threads)

2. **`partition_A(sA)`**:
   - Partitions SMEM tensor for this thread's reads
   - For WGMMA, **all threads in warp group access same SMEM**
   - Returns: `Tensor<Shape<16, 1, K, Stage>, smem>`

3. **`make_fragment_A(tCsA)`**:
   - Creates register fragment for loading
   - WGMMA doesn't use register operands for A/B!
   - Fragment used for: predicates, address computation
   - Returns: `Tensor<Shape<1, 1, K>, rmem>`

4. **`make_rmem_tensor(acc_shape, acc_dtype)`**:
   - Creates accumulator registers
   - Shape typically: `Shape<M_frags, N_frags, 1>`
   - Accumulator IS in registers (unlike A/B)

**CuTe C++ File**: [include/cute/atom/mma_traits_sm90_gmma.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/mma_traits_sm90_gmma.hpp)

---

### Kernel Frame 12: Main Computation Loop - DMA Warp

```python
# Line 724-782
is_dma_warp_group = warp_group_idx < self.num_dma_warp_groups

if warp_idx == self.load_warp_id:
    tile_sched = utils.StaticPersistentTileScheduler.create(
        tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
    )
    work_tile = tile_sched.initial_work_tile_info()

    mainloop_producer_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, self.ab_stage
    )

    while work_tile.is_valid_tile:
        tile_coord_mnl = work_tile.tile_idx
        tAgA_mkl = tAgA[(None, tile_coord_mnl[0], None, tile_coord_mnl[2])]
        tBgB_nkl = tBgB[(None, tile_coord_mnl[1], None, tile_coord_mnl[2])]

        mainloop_producer_state.reset_count()

        for k_tile in range(k_tile_cnt):
            # Wait for empty stage
            mainloop_pipeline.producer_acquire(mainloop_producer_state)

            # Slice to current k_tile
            tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
            tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]

            tBgB_k = tBgB_nkl[(None, mainloop_producer_state.count)]
            tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]

            # TMA load A/B
            cute.copy(
                tma_atom_a,
                tAgA_k,
                tAsA_pipe,
                tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                    mainloop_producer_state
                ),
                mcast_mask=a_mcast_mask,
            )
            cute.copy(...)

            mainloop_pipeline.producer_commit(mainloop_producer_state)
            mainloop_producer_state.advance()

        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()

    mainloop_pipeline.producer_tail(mainloop_producer_state)
```

**Unpacking Key Operations:**

**1. `StaticPersistentTileScheduler.create()`**:

```python
# cutlass/utils/__init__.py
class StaticPersistentTileScheduler:
    @staticmethod
    def create(params, block_idx, grid_dim):
        """Create tile scheduler for persistent kernel"""
        return TileScheduler(params, block_idx)

    def initial_work_tile_info(self):
        """Get first tile for this CTA"""
        return WorkTile(tile_idx=self.compute_initial_tile())

    def advance_to_next_work(self):
        """Move to next tile (persistent scheduling)"""
        self.current_tile += grid_dim
```

**Explanation**: Persistent kernels don't terminate after one tile. Each CTA processes multiple tiles in a loop, improving SM occupancy.

**2. `mainloop_pipeline.producer_acquire()`**:

```python
# cutlass/pipeline/sm90.py
class PipelineTmaAsync:
    def producer_acquire(self, state):
        """Wait for stage to become empty"""
        barrier_ptr = self.empty_barriers[state.index]
        cute.nvgpu.mbarrier.wait(barrier_ptr)
```

**MLIR Operations:**

```mlir
// Wait for empty barrier
%empty_barrier_ptr = ... : !llvm.ptr<i64, 3>
cute_nvgpu.mbarrier_wait %empty_barrier_ptr
  : (!llvm.ptr<i64, 3>) -> ()
```

**CuTe C++ Equivalent:**

```cpp
// Wait for stage to become available
CUTE_DEVICE void producer_acquire(PipelineState& state) {
  uint64_t* barrier = &empty_barriers[state.phase].barrier;

  // Wait for barrier to complete
  asm volatile(
    "{\n"
    ".reg .pred p;\n"
    "LAB_WAIT:\n"
    "  mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
    "  @!p bra.uni LAB_WAIT;\n"
    "}\n"
    :: "r"(barrier), "r"(state.parity)
  );
}
```

**3. TMA Copy with Barrier:**

```python
cute.copy(
    tma_atom_a,
    tAgA_k,           # Source: global memory
    tAsA_pipe,        # Destination: shared memory stage
    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(state),
    mcast_mask=a_mcast_mask,
)
```

**MLIR Operations:**

```mlir
%barrier_ptr = ... : !llvm.ptr<i64, 3>
%mcast_mask = ... : i16

cute_nvgpu.copy_tma_g2s
  %tma_atom, %src, %dst, %barrier_ptr, %mcast_mask
  : (!cute.copy_atom<tma>, !cute.tensor<...>, !cute.tensor<...>,
     !llvm.ptr<i64, 3>, i16) -> ()
```

**CuTe C++ Equivalent:**

```cpp
// TMA async copy from global to shared memory
template <class TMA, class GTensor, class STensor>
CUTE_DEVICE void copy_tma_g2s(
    TMA const& tma,
    GTensor const& g_tensor,
    STensor&& s_tensor,
    uint64_t* mbar_ptr,
    uint16_t mcast_mask) {
  if (threadIdx.x == 0) {
    // Issue TMA descriptor with barrier
    uint64_t tma_desc = reinterpret_cast<uint64_t>(&tma);
    uint64_t gmem_addr = reinterpret_cast<uint64_t>(g_tensor.data());
    uint32_t smem_addr = cast_smem_ptr_to_uint(s_tensor.data());

    if (mcast_mask != 0) {
      // Multicast version
      asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
        " [%0], [%1, {%2, %3}], [%4], %5;"
        :: "r"(smem_addr), "l"(tma_desc),
           "r"(coord_m), "r"(coord_k),
           "r"(mbar_ptr), "h"(mcast_mask)
      );
    } else {
      // Regular TMA
      asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(smem_addr), "l"(tma_desc),
           "r"(coord_m), "r"(coord_k), "r"(mbar_ptr)
      );
    }
  }
}
```

**Explanation:**

- **Single thread issues TMA**: Only thread 0 needs to issue (hardware handles broadcast)
- **Barrier integration**: TMA auto-arrives on barrier when transfer completes
- **Multicast**: With mask, hardware sends to multiple CTAs in cluster
- **Transaction count**: TMA decrements barrier by number of bytes transferred

---

### Kernel Frame 13: Main Computation Loop - MMA Warp

```python
# Line 785-916
if not is_dma_warp_group:
    tile_sched = utils.StaticPersistentTileScheduler.create(...)
    work_tile = tile_sched.initial_work_tile_info()

    mainloop_consumer_read_state = pipeline.make_pipeline_state(...)
    mainloop_consumer_release_state = pipeline.make_pipeline_state(...)

    while work_tile.is_valid_tile:
        tile_coord_mnl = work_tile.tile_idx
        gC_mnl_slice = gC_mnl[(None, None, *tile_coord_mnl)]

        # Initialize accumulators
        accumulators.fill(0.0)
        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
        cute.nvgpu.warpgroup.fence()

        # Prologue MMAs
        for k_tile in range(prologue_mma_cnt):
            mainloop_pipeline.consumer_wait(mainloop_consumer_read_state)

            for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                k_block_coord = (None, None, k_block_idx,
                                mainloop_consumer_read_state.index)
                cute.gemm(
                    tiled_mma,
                    accumulators,
                    tCrA[k_block_coord],
                    tCrB[k_block_coord],
                    accumulators,
                )

            cute.nvgpu.warpgroup.commit_group()
            mainloop_consumer_read_state.advance()

        # Steady-state MMAs
        for k_tile in range(prologue_mma_cnt, k_tile_cnt):
            mainloop_pipeline.consumer_wait(mainloop_consumer_read_state)

            for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                k_block_coord = (...)
                cute.gemm(tiled_mma, accumulators,
                         tCrA[k_block_coord], tCrB[k_block_coord], accumulators)

            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(k_pipe_mmas)

            mainloop_pipeline.consumer_release(mainloop_consumer_release_state)
            mainloop_consumer_release_state.advance()
            mainloop_consumer_read_state.advance()

        # Epilogue...
```

**Unpacking Key MMA Operations:**

**1. `tiled_mma.set()` and `fence()`:**

```python
# Set accumulation mode for WGMMA
tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
cute.nvgpu.warpgroup.fence()
```

**MLIR Operations:**

```mlir
cute_nvgpu.wgmma_set_accumulate : () -> ()
cute_nvgpu.wgmma_fence_aligned : () -> ()
```

**CuTe C++ Equivalent:**

```cpp
// Set WGMMA to accumulation mode (not overwrite)
CUTE_DEVICE void wgmma_set_accumulate() {
  asm volatile("wgmma.fence.sync.aligned;");
  asm volatile(
    "{"
    ".reg .b32 status;"
    "mov.u32 status, %0;"
    "setmaxnreg.dec.sync.aligned.u32 status;"
    "}"
    :: "n"(240)  // Use 240 registers for MMA
  );
}

// Fence: Ensure all prior WGMMA setup completes
CUTE_DEVICE void wgmma_fence_aligned() {
  asm volatile("wgmma.fence.sync.aligned;");
}
```

**Explanation:**
- **ACCUMULATE mode**: WGMMA adds to existing accumulator (D = A*B + C)
- **Fence**: Synchronizes WGMMA setup across warp group

**2. `cute.gemm()` - The Core WGMMA Operation:**

```python
cute.gemm(
    tiled_mma,      # TiledMMA config
    accumulators,   # D (output/accumulator)
    tCrA[...],      # A (SMEM)
    tCrB[...],      # B (SMEM)
    accumulators,   # C (input accumulator)
)
```

**MLIR Operations:**

```mlir
%d_new = cute_nvgpu.wgmma
         %tiled_mma, %c_acc, %a_smem, %b_smem
         : (!cute.tiled_mma<...>,
            !cute.tensor<..., rmem>,   // Accumulator
            !cute.tensor<..., smem>,   // A
            !cute.tensor<..., smem>)   // B
         -> !cute.tensor<..., rmem>    // New accumulator
```

**CuTe C++ Equivalent:**

```cpp
// include/cute/arch/mma_sm90_gmma.hpp
template <class TiledMMA, class TD, class TA, class TB, class TC>
CUTE_DEVICE void gemm(
    TiledMMA const& mma,
    TD& d,  // Output accumulator
    TA const& a,  // A in SMEM
    TB const& b,  // B in SMEM
    TC const& c)  // Input accumulator
{
  // For SM90_64x128x16_F32F16F16_SS:
  // - A: 64x16 tile in SMEM
  // - B: 128x16 tile in SMEM
  // - C/D: 64x128 accumulator in registers (FP32)

  uint64_t desc_a = make_smem_desc(a);
  uint64_t desc_b = make_smem_desc(b);

  // Issue WGMMA instruction
  asm volatile(
    "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
    "{%0, %1, %2, %3},"  // Output: 4 registers (64x128 / 32 threads)
    "%4,"                 // Descriptor A
    "%5,"                 // Descriptor B
    "1;"                  // Scale D by 1 (accumulate mode)
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
    : "l"(desc_a), "l"(desc_b)
  );
}
```

**WGMMA Instruction Breakdown:**

```
wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16
       ^^^^^  ^^^^  ^^^^^^^  ^^^^^^^^^^^  ^^^^^^^^^^^^
       |      |     |        |            |
       |      |     |        |            Operand types: f32(out) f16(A) f16(B)
       |      |     |        Tile shape: M=64, N=128, K=16
       |      |     Aligned memory access
       |      Synchronous within warp group
       Asynchronous across warp groups
```

**Key Properties:**
- **Operates on warp group** (128 threads = 4 warps)
- **Reads from SMEM** (via descriptors), not registers
- **Writes to registers** (accumulator distributed across warp group)
- **Async**: Can pipeline multiple WGMMAs

**3. `commit_group()` and `wait_group()`:**

```python
cute.nvgpu.warpgroup.commit_group()
cute.nvgpu.warpgroup.wait_group(k_pipe_mmas)
```

**MLIR Operations:**

```mlir
cute_nvgpu.wgmma_commit_group : () -> ()
cute_nvgpu.wgmma_wait_group %n : (i32) -> ()
```

**CuTe C++ Equivalent:**

```cpp
// Commit issued WGMMA instructions
CUTE_DEVICE void wgmma_commit_group() {
  asm volatile("wgmma.commit_group.sync.aligned;");
}

// Wait for all but N WGMMA groups to complete
CUTE_DEVICE void wgmma_wait_group(int n) {
  asm volatile(
    "wgmma.wait_group.sync.aligned %0;"
    :: "n"(n)
  );
}

// Example: Software pipeline with 2 stages
for (int k = 0; k < K; ++k) {
  // Issue WGMMA for k+1
  wgmma(..., smem[(k+1) % 2]);
  commit_group();

  // Wait for k-1 to complete (keep 1 in flight)
  wait_group(1);

  // Release smem[k % 2]
  pipeline.release(k % 2);
}
```

**Explanation:**

- **`commit_group()`**: Marks end of WGMMA group (can have multiple WGMMAs per group)
- **`wait_group(N)`**: Waits until at most N groups are in flight
- **Software Pipelining**: Overlap WGMMA with memory loads:
  ```
  Iteration k:
    1. Wait for stage k to be full (data ready)
    2. Issue WGMMA(k) and commit
    3. Wait for WGMMA(k-2) to complete
    4. Release stage k-1 (data consumed)
  ```

**CuTe C++ File**: [include/cute/arch/mma_sm90_gmma.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm90_gmma.hpp)

---

### Kernel Frame 14: Epilogue - Register to SMEM Copy

```python
# Line 918-959
# Partition for epilogue
copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
    self.c_layout, elem_ty_d=self.c_dtype, elem_ty_acc=self.acc_dtype
)

copy_atom_C = cute.make_copy_atom(
    cute.nvgpu.warp.StMatrix8x8x16bOp(
        self.c_layout.is_m_major_c(), 4,
    ),
    self.c_dtype,
)

tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_Atom)

thr_copy_r2s = tiled_copy_r2s.get_slice(
    tidx - self.num_dma_warp_groups * self.num_threads_per_warp_group
)
tRS_sD = thr_copy_r2s.partition_D(sC)
tRS_rAcc = tiled_copy_r2s.retile(accumulators)

# For each epilogue tile:
for epi_idx in cutlass.range_constexpr(epi_tile_num):
    # Copy from accumulators to D registers
    for epi_v in cutlass.range_constexpr(size_tRS_rD):
        tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]

    # Type conversion
    acc_vec = tRS_rD.load()
    tRS_rD_out.store(acc_vec.to(self.c_dtype))

    # Copy from D registers to shared memory
    epi_buffer = (num_prev_epi_tiles + epi_idx) % cute.size(tRS_sD, mode=[3])
    cute.copy(tiled_copy_r2s, tRS_rD_out,
              tRS_sD[(None, None, None, epi_buffer)])

    cute.arch.fence_proxy(
        cute.arch.ProxyKind.async_shared,
        space=cute.arch.SharedSpace.shared_cta,
    )
    self.epilog_sync_barrier.arrive_and_wait()
```

**Unpacking Epilogue Copy Operations:**

**1. `make_copy_atom()` - StMatrix:**

```python
copy_atom_C = cute.make_copy_atom(
    cute.nvgpu.warp.StMatrix8x8x16bOp(is_m_major, num_matrices=4),
    self.c_dtype,
)
```

**Explanation:**

Hopper's `stmatrix` instruction stores a **matrix fragment** from registers to SMEM in one instruction:

```
stmatrix.sync.aligned.m8n8.x4.shared.b16 [smem_addr], {r0, r1, r2, r3};
                             ^^^                       ^^^^^^^^^^^^^^^
                             4 matrices (8x8 each)     4 registers
```

**MLIR Operations:**

```mlir
%copy_atom = cute.make_copy_atom
             !cute_nvgpu.stmatrix<m8n8, x4, b16, m_major>
             : () -> !cute.copy_atom<stmatrix>
```

**CuTe C++ Equivalent:**

```cpp
// include/cute/arch/copy_sm90.hpp
struct SM90_U16x8_STSM_N {
  // Store 8x8 tile (16-bit elements) to SMEM using stmatrix
  template <class TS, class SrcLayout, class DstLayout>
  CUTE_DEVICE void copy(
      TS const& src,  // Source in registers
      TS* dst)        // Destination in SMEM
  {
    uint32_t smem_addr = cast_smem_ptr_to_uint(dst);

    asm volatile(
      "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};"
      :: "r"(smem_addr),
         "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3])
    );
  }
};

using StMatrix_Atom = Copy_Atom<SM90_U16x8_STSM_N>;
```

**2. `make_tiled_copy_C_atom()` and `make_tiled_copy_S()`:**

These create **tiled copy** operations that handle distribution of accumulator across threads.

```python
tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_Atom)
```

**CuTe C++ Equivalent:**

```cpp
// Create tiled copy matching MMA accumulator layout
template <class CopyAtom, class TiledMMA>
auto make_tiled_copy_C(CopyAtom const& atom, TiledMMA const& mma) {
  // Match copy layout to MMA's C layout
  auto mma_c_layout = mma.layout_C();

  return make_tiled_copy(
    atom,
    mma_c_layout.layout_tv(),  // Thread-value layout
    make_layout(make_shape(Int<8>{}, Int<8>{}))  // Tile shape for stmatrix
  );
}
```

**3. Fence and Barrier:**

```python
cute.arch.fence_proxy(
    cute.arch.ProxyKind.async_shared,
    space=cute.arch.SharedSpace.shared_cta,
)
self.epilog_sync_barrier.arrive_and_wait()
```

**MLIR Operations:**

```mlir
cute_nvgpu.fence_proxy_async_shared : () -> ()
cute_nvgpu.mbarrier_arrive_and_wait %barrier_ptr : (!llvm.ptr<i64, 3>) -> ()
```

**CuTe C++ Equivalent:**

```cpp
// Fence async SMEM operations
CUTE_DEVICE void fence_proxy_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;");
}

// Named barrier: arrive and wait
CUTE_DEVICE void arrive_and_wait(uint64_t* barrier) {
  asm volatile(
    "{\n"
    ".reg .pred p;\n"
    ".reg .b32 state;\n"
    "LAB_WAIT:\n"
    "  mbarrier.arrive_expect_tx.shared.b64 state, [%0], 0;\n"
    "  mbarrier.try_wait.parity.shared.b64 p, [%0], state;\n"
    "  @!p bra.uni LAB_WAIT;\n"
    "}\n"
    :: "r"(barrier)
  );
}
```

**Explanation:**

- **Fence**: Ensures all prior async SMEM writes complete before proceeding
- **Named Barrier**: Synchronizes MMA warps before TMA store
  - All MMA threads must finish writing to SMEM
  - Then single thread can issue TMA store

---

### Kernel Frame 15: Epilogue - SMEM to GMEM TMA Store

```python
# Line 961-972
# Partition for TMA store
bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
    tma_atom_c,
    0,
    cute.make_layout(1),
    cute.group_modes(sC, 0, 2),
    tCgC_for_tma_partition,
)

# Copy from shared memory to global memory
if warp_idx == self.epi_store_warp_id:
    cute.copy(
        tma_atom_c,
        bSG_sD[(None, epi_buffer)],
        bSG_gD[(None, gmem_coord)],
    )
    tma_store_pipeline.producer_commit()
    tma_store_pipeline.producer_acquire()
```

**Unpacking TMA Store:**

TMA can **store** from SMEM to GMEM (reverse of load).

**MLIR Operations:**

```mlir
// TMA store: SMEM → GMEM
cute_nvgpu.copy_tma_s2g
  %tma_atom, %src_smem, %dst_gmem
  : (!cute.copy_atom<tma>, !cute.tensor<..., smem>, !cute.tensor<..., gmem>)
  -> ()
```

**CuTe C++ Equivalent:**

```cpp
// TMA store from SMEM to GMEM
template <class TMA, class STensor, class GTensor>
CUTE_DEVICE void copy_tma_s2g(
    TMA const& tma,
    STensor const& s_tensor,
    GTensor&& g_tensor)
{
  if (threadIdx.x == 0) {
    uint64_t tma_desc = reinterpret_cast<uint64_t>(&tma);
    uint32_t smem_addr = cast_smem_ptr_to_uint(s_tensor.data());
    uint64_t gmem_addr = reinterpret_cast<uint64_t>(g_tensor.data());

    asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
      " [%0, {%1, %2}], [%3];"
      :: "l"(tma_desc),
         "r"(coord_m), "r"(coord_n),
         "r"(smem_addr)
    );
  }
}
```

**TMA Store Pipeline:**

```python
tma_store_pipeline.producer_commit()
tma_store_pipeline.producer_acquire()
```

This implements **pipelined epilogue**:
1. **Commit**: Mark TMA store issued
2. **Acquire**: Wait for previous store to complete before reusing SMEM stage

Similar to load pipeline, but simpler (no barriers needed for SMEM→GMEM).

---

## MLIR Operations → CuTe C++ Mapping

This section provides a comprehensive mapping table from Python/MLIR operations to their C++ equivalents.

### Core Types

| Python/MLIR Type | CuTe C++ Type | File |
|-----------------|---------------|------|
| `cute.Tensor` | `Tensor<Engine, Layout>` | [tensor.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/tensor.hpp) |
| `cute.Layout` | `Layout<Shape, Stride>` | [layout.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/layout.hpp) |
| `cute.ComposedLayout` | `ComposedLayout<...>` | [layout_composed.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/layout_composed.hpp) |
| `cute.TiledMma` | `TiledMMA<Atom, Layout, Tiler>` | [mma_traits.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/mma_traits.hpp) |
| `cute.CopyAtom` | `Copy_Atom<Traits>` | [copy_atom.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/copy_atom.hpp) |
| `cute.Swizzle` | `Swizzle<Bits, Base, Shift>` | [swizzle.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/swizzle.hpp) |
| `!cute.ptr<T, space, align>` | `T*` (with address space) | [pointer.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/pointer.hpp) |

### Shape and Layout Operations

| Python/MLIR Operation | CuTe C++ Function | File |
|-----------------------|-------------------|------|
| `cute.make_layout(shape, stride)` | `make_layout(shape, stride)` | [layout.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/layout.hpp) |
| `cute.size(layout, mode)` | `size<Mode>(layout)` | [int_tuple.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/numeric/int_tuple.hpp) |
| `cute.cosize(layout)` | `cosize(layout)` | [layout.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/layout.hpp) |
| `cute.slice_(layout, coord)` | `slice(layout, coord)` | [tuple_algorithms.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/algorithm/tuple_algorithms.hpp) |
| `cute.append(shape, val, up_to_rank)` | `append(shape, val, up_to_rank)` | [tuple_algorithms.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/algorithm/tuple_algorithms.hpp) |
| `cute.composition(f, layout)` | `composition(f, layout)` | [layout_composed.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/layout_composed.hpp) |
| `cute.tile_to_shape(layout, shape, order)` | `tile_to_shape(layout, shape, order)` | [layout.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/layout.hpp) |

### Tensor Operations

| Python/MLIR Operation | CuTe C++ Function | File |
|-----------------------|-------------------|------|
| `cute.make_tensor(ptr, layout)` | `make_tensor(ptr, layout)` | [tensor.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/tensor.hpp) |
| `cute.make_rmem_tensor(shape, dtype)` | `make_tensor<rmem>(layout, dtype)` | [tensor.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/tensor.hpp) |
| `cute.local_tile(tensor, tile, coord)` | `local_tile(tensor, tile, coord)` | [tensor.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/tensor.hpp) |
| `cute.zipped_divide(tensor, tiler)` | `zipped_divide(tensor, tiler)` | [tensor.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/tensor.hpp) |
| `cute.group_modes(tensor, begin, end)` | `group_modes<Begin, End>(tensor)` | [tuple_algorithms.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/algorithm/tuple_algorithms.hpp) |
| `tensor.load()` | `tensor(coord)` or `*tensor.data()` | [tensor.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/tensor.hpp) |
| `tensor.store(data)` | `tensor(coord) = data` | [tensor.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/tensor.hpp) |

### Copy Operations

| Python/MLIR Operation | CuTe C++ Function | File |
|-----------------------|-------------------|------|
| `cute.copy(atom, src, dst)` | `copy(atom, src, dst)` | [copy.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/algorithm/copy.hpp) |
| `cute.make_copy_atom(op, dtype)` | `Copy_Atom<Traits>{}` | [copy_atom.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/copy_atom.hpp) |
| `cute.make_tiled_copy(atom, layout, tiler)` | `make_tiled_copy(atom, layout, tiler)` | [copy.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/algorithm/copy.hpp) |
| `tiled_copy.get_slice(thread_idx)` | `get_slice(thread_idx, tiled_copy)` | [copy.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/algorithm/copy.hpp) |
| `thr_copy.partition_S(tensor)` | `thr_copy.partition_S(tensor)` | [copy.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/algorithm/copy.hpp) |
| `thr_copy.partition_D(tensor)` | `thr_copy.partition_D(tensor)` | [copy.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/algorithm/copy.hpp) |

### TMA Operations (Hopper)

| Python/MLIR Operation | CuTe C++ Function | File |
|-----------------------|-------------------|------|
| `cute.nvgpu.cpasync.make_tiled_tma_atom(op, tensor, layout, tile)` | `make_tma_copy(op, tensor, layout, tile)` | [copy_sm90_tma.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm90_tma.hpp) |
| `cute.nvgpu.cpasync.tma_partition(atom, cta, layout, s, g)` | `tma_partition(atom, cta, layout, s, g)` | [copy.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/algorithm/copy.hpp) |
| `cute.nvgpu.cpasync.prefetch_descriptor(atom)` | `prefetch_tma_descriptor(atom)` | [copy_sm90_desc.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm90_desc.hpp) |
| `cute.copy(tma_atom, src, dst, tma_bar_ptr, mcast_mask)` | `copy(tma_atom, src, dst, bar, mask)` | [copy_sm90_tma.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm90_tma.hpp) |

### MMA Operations

| Python/MLIR Operation | CuTe C++ Function | File |
|-----------------------|-------------------|------|
| `cute.make_tiled_mma(atom, layout, tiler)` | `make_tiled_mma(atom, layout, tiler)` | [gemm.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/algorithm/gemm.hpp) |
| `tiled_mma.get_slice(thread_idx)` | `get_slice(thread_idx, tiled_mma)` | [mma_traits.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/mma_traits.hpp) |
| `thr_mma.partition_A(tensor)` | `thr_mma.partition_A(tensor)` | [mma_traits.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/mma_traits.hpp) |
| `tiled_mma.make_fragment_A(partition)` | `make_fragment_A(partition, mma)` | [mma_traits.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/mma_traits.hpp) |
| `cute.gemm(mma, D, A, B, C)` | `gemm(mma, D, A, B, C)` | [gemm.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/algorithm/gemm.hpp) |

### WGMMA Operations (Hopper)

| Python/MLIR Operation | CuTe C++ Function | File |
|-----------------------|-------------------|------|
| `cute.nvgpu.warpgroup.fence()` | `wgmma_fence_aligned()` | [mma_sm90_gmma.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm90_gmma.hpp) |
| `cute.nvgpu.warpgroup.commit_group()` | `wgmma_commit_group()` | [mma_sm90_gmma.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm90_gmma.hpp) |
| `cute.nvgpu.warpgroup.wait_group(n)` | `wgmma_wait_group<N>()` | [mma_sm90_gmma.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm90_gmma.hpp) |
| `tiled_mma.set(Field.ACCUMULATE, True)` | `wgmma_set_accumulate()` | [mma_sm90_gmma.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm90_gmma.hpp) |
| `cute.nvgpu.warpgroup.make_smem_layout_atom(swizzle, dtype)` | `make_smem_layout_atom<Swizzle>(dtype)` | [copy_traits_sm90.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/copy_traits_sm90.hpp) |

### Synchronization Operations

| Python/MLIR Operation | CuTe C++ Function | File |
|-----------------------|-------------------|------|
| `cute.arch.syncthreads()` | `__syncthreads()` | Standard CUDA |
| `cute.arch.cluster_arrive()` | `cluster_arrive()` | [cluster_sm90.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/cluster_sm90.hpp) |
| `cute.arch.cluster_wait()` | `cluster_wait()` | [cluster_sm90.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/cluster_sm90.hpp) |
| `cute.nvgpu.mbarrier.init(ptr, count, tx)` | `mbarrier_init(ptr, count, tx)` | [copy_sm90_tma.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm90_tma.hpp) |
| `cute.nvgpu.mbarrier.wait(ptr)` | `mbarrier_wait(ptr)` | [copy_sm90_tma.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm90_tma.hpp) |
| `cute.nvgpu.mbarrier.arrive(ptr)` | `mbarrier_arrive(ptr)` | [copy_sm90_tma.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm90_tma.hpp) |

### Thread/Block Operations

| Python/MLIR Operation | CuTe C++ Function | File |
|-----------------------|-------------------|------|
| `cute.arch.thread_idx()` | `thread_idx()` | [util.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/util.hpp) |
| `cute.arch.block_idx()` | `blockIdx.x/y/z` | Standard CUDA |
| `cute.arch.warp_idx()` | `threadIdx.x / 32` | [util.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/util.hpp) |
| `cute.arch.block_idx_in_cluster()` | `block_idx_in_cluster()` | [cluster_sm90.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/cluster_sm90.hpp) |
| `cute.arch.make_warp_uniform(val)` | `__shfl_sync(0xffffffff, val, 0)` | [util.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/util.hpp) |
| `cute.arch.warpgroup_reg_alloc(n)` | `setmaxnreg.inc.sync.aligned.u32` | [mma_sm90_gmma.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm90_gmma.hpp) |
| `cute.arch.warpgroup_reg_dealloc(n)` | `setmaxnreg.dec.sync.aligned.u32` | [mma_sm90_gmma.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm90_gmma.hpp) |

### Pipeline Operations

| Python/MLIR Operation | CuTe C++ Function | File |
|-----------------------|-------------------|------|
| `pipeline.PipelineTmaAsync.create(...)` | `PipelineTmaAsync<Stages>{...}` | Custom (see examples) |
| `pipeline.producer_acquire(state)` | `producer_acquire(state)` | Custom (see examples) |
| `pipeline.producer_commit(state)` | `producer_commit(state)` | Custom (see examples) |
| `pipeline.consumer_wait(state)` | `consumer_wait(state)` | Custom (see examples) |
| `pipeline.consumer_release(state)` | `consumer_release(state)` | Custom (see examples) |

---

## Key Conceptual Mappings

### 1. **Lazy vs Eager Evaluation**

**Python/MLIR (Lazy)**:
```python
layout = cute.make_layout((128, 256))  # Builds IR, doesn't execute
tensor = cute.make_tensor(ptr, layout)  # Builds IR
result = tensor.load()  # Builds IR
# Execution happens at kernel launch
```

**C++ (Eager)**:
```cpp
auto layout = make_layout(Shape<_128, _256>{});  // Compile-time type
auto tensor = make_tensor(ptr, layout);  // Runtime pointer + compile-time layout
auto result = copy(tensor, ...);  # Immediate execution (at runtime)
```

### 2. **Type System**

**Python/MLIR**:
- Runtime types in MLIR: `!cute.layout<...>`, `!cute.tensor<...>`
- Type checking at MLIR compilation, not Python
- Some compile-time via `cutlass.Constexpr`

**C++ CuTe**:
- Compile-time types: `Layout<Shape<_128, _256>, Stride<_256, _1>>`
- Template metaprogramming for layout transformations
- Zero runtime overhead for layout calculations

### 3. **Memory Spaces**

**Python/MLIR**:
```python
ptr = cute.make_ptr(dtype, addr, cute.AddressSpace.smem)
```

**C++**:
```cpp
__shared__ float smem[1024];
auto ptr = make_smem_ptr(smem);
// Or: float* gmem_ptr (implicit global)
```

### 4. **Swizzling**

**Python/MLIR**:
```python
swizzle = cute.make_swizzle(128, 4, 3)
composed = cute.composition(swizzle, layout)
```

**C++**:
```cpp
using Swizzle = Swizzle<3, 4, 3>;  // <log2(Base), M, S>
auto composed = composition(Swizzle{}, layout);
```

### 5. **Pipeline State Management**

**Python/MLIR**:
```python
# Explicit state objects
state = pipeline.make_pipeline_state(PipelineUserType.Producer, num_stages)
pipeline.producer_acquire(state)
state.advance()
```

**C++**:
```cpp
// Manually track phase/index
struct PipelineState {
  int phase;
  int index;
  int count;
};
PipelineState state{0, 0, 0};
producer_acquire(state);
state = advance(state);
```

---

## Summary: Python → MLIR → C++ Flow

1. **Python DSL Code**:
   ```python
   @cute.jit
   def gemm(a, b, c):
       layout = cute.make_layout((128, 256))
       tensor = cute.make_tensor(ptr, layout)
   ```

2. **MLIR Generation** (via `@cute.jit`):
   ```mlir
   func.func @gemm(%a: !cute.tensor<...>, ...) {
     %layout = cute.make_layout : () -> !cute.layout<...>
     %tensor = cute.make_view %ptr, %layout : (...) -> !cute.tensor<...>
   }
   ```

3. **MLIR Lowering**:
   - `cute` dialect → `gpu` dialect → `llvm` dialect → `nvvm` dialect

4. **PTX Generation**:
   ```ptx
   .entry gemm(...) {
     // PTX instructions
     cp.async.bulk.tensor.2d.shared::cluster.global ...
     wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 ...
   }
   ```

5. **Conceptual C++ Equivalent**:
   ```cpp
   template <class A, class B, class C>
   __global__ void gemm(A a, B b, C c) {
     auto layout = Layout<Shape<_128, _256>>{};
     auto tensor = make_tensor(ptr, layout);
   }
   ```

The key insight: **CuTe DSL generates the same PTX as hand-written CuTe C++**, but allows Python flexibility for rapid prototyping while maintaining C++ performance.

---

## Additional Resources

### CuTe C++ Core Files

- **Layout**: [include/cute/layout.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/layout.hpp)
- **Tensor**: [include/cute/tensor.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/tensor.hpp)
- **Copy**: [include/cute/algorithm/copy.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/algorithm/copy.hpp)
- **GEMM**: [include/cute/algorithm/gemm.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/algorithm/gemm.hpp)
- **Swizzle**: [include/cute/swizzle.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/swizzle.hpp)

### Hopper-Specific Files

- **TMA**: [include/cute/arch/copy_sm90_tma.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm90_tma.hpp)
- **WGMMA**: [include/cute/arch/mma_sm90_gmma.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm90_gmma.hpp)
- **Cluster**: [include/cute/arch/cluster_sm90.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/cluster_sm90.hpp)
- **Barriers**: [include/cute/arch/copy_sm90_tma.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm90_tma.hpp)

### CUTLASS Documentation

- **CuTe Tutorial**: https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md
- **CUTLASS 3.x**: https://github.com/NVIDIA/cutlass/blob/main/media/docs/cutlass_3x.md
- **Hopper GEMM**: https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/hopper_gemm_with_tma.cu

---

## Conclusion

This document provides a comprehensive frame-by-frame trace of CuTeDSL's Python code, showing:

1. **How Python methods map to MLIR operations**
2. **How MLIR operations lower to PTX**
3. **How each concept maps to CuTe C++ equivalents**

The key takeaway: **CuTe DSL is not an abstraction layer** - it's a **direct Python interface** to the same CuTe abstractions used in C++. The MLIR IR serves as a thin translation layer that generates identical PTX to hand-written C++ CuTe code.

This enables:
- **Rapid prototyping** in Python
- **Same performance** as C++ (identical PTX)
- **Easy debugging** (Python stack traces map to MLIR ops)
- **Seamless interop** (concepts translate 1:1 between Python and C++)
