#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>
#include <pybind11/pybind11.h>

// Upstream symbol implemented in the copied GroupMM.cu (unchanged).

void bf16bf16_grouped_mm(
    at::Tensor mat_a,                          // bf16
    at::Tensor mat_b,                          // bf16
    c10::optional<at::Tensor> offs,            // optional prefix-sum/group metadata
    c10::optional<at::Tensor> bias,            // optional bf16 bias
    at::Tensor& out);                          // bf16 out (preallocated)

// // -------- existing preallocated variant (kept) --------
// static at::Tensor _grouped_mm_out(
//     const at::Tensor& mat_a,
//     const at::Tensor& mat_b,
//     c10::optional<at::Tensor> offs,
//     c10::optional<at::Tensor> bias,
//     at::Tensor out) {
//   TORCH_CHECK(mat_a.is_cuda() && mat_b.is_cuda() && out.is_cuda(),
//               "groupedmm_ext: tensors must be CUDA");
//   TORCH_CHECK(mat_a.scalar_type() == at::kBFloat16 &&
//               mat_b.scalar_type() == at::kBFloat16 &&
//               out.scalar_type()   == at::kBFloat16,
//               "groupedmm_ext: bf16 only (mirrors upstream)");
//   at::cuda::detail::bf16bf16_grouped_mm(mat_a, mat_b, offs, bias, out);
//   return out;
// }

// -------- new convenience allocator variant --------
//
// Mirrors ATen BLAS helpers' pattern: infer output sizes from inputs,
// allocate a row-major bf16 output, then call the same upstream kernel.
// See aten/native/cuda/Blas.cpp for the general approach. :contentReference[oaicite:2]{index=2}
 at::Tensor create_grouped_gemm_output_tensor(const Tensor& mat_a,
  const Tensor& mat_b,
  const std::optional<at::Tensor>& offs,
  std::optional<c10::ScalarType> out_dtype
  ) {
    c10::SmallVector<int64_t, 3> out_size;
    const bool a_is_2d = mat_a.dim() == 2;
    const bool b_is_2d = mat_b.dim() == 2;
    if (a_is_2d) {
      if (b_is_2d) {
        out_size = {offs->size(0), mat_a.size(0), mat_b.size(1)};
      } else {
        TORCH_CHECK(offs->size(0) == mat_b.size(0), "matrix batch sizes have to match");
        out_size = {mat_a.size(0), mat_b.size(-1)};
      }
    } else {
      if (b_is_2d) {
        // this case is not actually encountered for MoE gemms
        TORCH_CHECK(offs->size(0) == mat_a.size(0), "matrix batch sizes have to match");
        out_size = {mat_a.size(1), mat_b.size(1)};
      } else { // regular bmm
        TORCH_CHECK(mat_a.size(0) == mat_b.size(0), "batched dimension has to match");
        out_size = {mat_a.size(0), mat_a.size(1), mat_b.size(-1)};
      }
    }

    const auto out_dtype_ = out_dtype.value_or(kBFloat16);
    TORCH_CHECK(out_dtype_ == kBFloat16, "Only bf16 high precision output types are supported for grouped gemm");

    // For TMA transfers, strides of output tensor have to be either
    // 1, or aligned to 16 bytes.
    const auto last_dim = out_size.size() - 1;
    const auto alignment = 16 / c10::elementSize(out_dtype_);
    const int64_t size_padded = (out_size[last_dim] + alignment - 1) / alignment * alignment;
    std::vector<int64_t> out_stride;
    if (a_is_2d != b_is_2d) {
      out_stride = {size_padded, 1};
    } else {
      out_stride = {out_size[1] * size_padded, size_padded, 1};
    }
    std::cout << "Allocated out size: " << out_size << "\n";
    return at::empty_strided(out_size, out_stride, mat_a.options().dtype(out_dtype_));
  }

bool check_valid_strides_and_return_transposed(const Tensor& mat) {
    IntArrayRef tensor_strides = mat.strides();
    IntArrayRef tensor_sizes = mat.sizes();
    int end_dim = mat.dim() - 1;
    int alignment = 16 / mat.element_size();
    TORCH_CHECK(uint64_t(mat.data_ptr()) % 16 ==0, "expected data_ptr to be aligned to 16 bytes\n");
    if ((tensor_strides[end_dim - 1] == 1) && (tensor_strides[end_dim] >= std::max<int64_t>(1, tensor_sizes[end_dim - 1]))) {
      TORCH_CHECK(tensor_strides[end_dim] % alignment == 0, "strides should be multiple of 16 bytes");
      return true;
    } else if ((tensor_strides[end_dim] == 1) && (tensor_strides[end_dim - 1] >= std::max<int64_t>(1, tensor_sizes[end_dim]))) {
      TORCH_CHECK(tensor_strides[end_dim - 1] % alignment == 0, "strides should be multiple of 16 bytes");
      return false;
    } else {
      TORCH_CHECK(false, "Invalid strides/sizes, got ", mat.strides(), " for strides and ", mat.sizes(), " for sizes");
    }
  }

  static at::Tensor _grouped_mm(
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    c10::optional<at::Tensor> offs,
    c10::optional<at::Tensor> bias) {

  // TORCH_CHECK(mat_a.is_cuda() && mat_b.is_cuda(),
  //             "groupedmm_ext: tensors must be CUDA");
  // TORCH_CHECK(mat_a.scalar_type() == at::kBFloat16 &&
  //             mat_b.scalar_type() == at::kBFloat16,
  //             "groupedmm_ext: bf16 only (mirrors upstream)");

  // // Infer output shape:
  // // Output is always 2D, with rows equal to "total M" and columns equal to N.
  // // - If A is 3D [G, M, K], rows = G*M
  // // - If A is 2D [sum_m, K], rows = sum_m
  // int64_t rows;
  // if (mat_a.dim() == 3) {
  //   TORCH_CHECK(mat_a.size(-2) >= 0, "invalid A shape");
  //   rows = mat_a.size(0) * mat_a.size(1);
  // } else {
  //   TORCH_CHECK(mat_a.dim() == 2, "A must be 2D or 3D");
  //   rows = mat_a.size(0);
  // }

  // // Columns (N) come from B’s “N” dim. For 2D B, N = B.size(1).
  // // For 3D B, we must respect layout:
  // //   row-major B  (fastest dim = last):   shape [G, K, N] => N = size(2)
  // //   column-major B (fastest dim = -2):   shape [G, N, K] => N = size(1)
  // int64_t cols;
  // if (mat_b.dim() == 2) {
  //   cols = mat_b.size(1);
  // } else {
  //   TORCH_CHECK(mat_b.dim() == 3, "B must be 2D or 3D");
  //   const bool b_row_major = (mat_b.stride(-1) == 1);
  //   cols = b_row_major ? mat_b.size(2) : mat_b.size(1);
  // }

  // // If BOTH inputs are 2D, upstream requires offs to define grouping/shape.
  // // See GroupMM.cu: it reads offs->size(0) in this case. :contentReference[oaicite:3]{index=3}
  // if (mat_a.dim() == 2 && mat_b.dim() == 2) {
  //   TORCH_CHECK(offs.has_value(),
  //     "groupedmm_ext._grouped_mm: when A and B are both 2D (ragged case), "
  //     "'offs' must be provided to describe groups.");
  // }

  // // Allocate output row-major bf16 on the same device as A.
  // auto out = at::empty({rows, cols}, mat_a.options().dtype(at::kBFloat16));

  TORCH_CHECK(mat_a.dtype() == at::kBFloat16, "Expected mat_a to be BFloat16 matrix got ", mat_a.scalar_type());
  TORCH_CHECK(mat_b.dtype() == at::kBFloat16, "Expected mat_a to be BFloat16 matrix got ", mat_b.scalar_type());
  TORCH_CHECK(mat_a.dim() == 2 || mat_a.dim() == 3, "mat_a has to be 2 or 3d");
  TORCH_CHECK(mat_b.dim() == 2 || mat_b.dim() == 3, "mat_b has to be 2 or 3d");
  const bool a_is_2d = mat_a.dim() == 2;
  const bool b_is_2d = mat_b.dim() == 2;
 
  if (!a_is_2d || !b_is_2d) {
    TORCH_CHECK(mat_a.size(-1) == mat_b.size(-2), "contraction dimension of mat_a and mat_b must match");
  }

  // check that the strides are valid, the fn will throw an error if not
  check_valid_strides_and_return_transposed(mat_a);
  check_valid_strides_and_return_transposed(mat_b);
  TORCH_CHECK(offs.has_value() ==  (a_is_2d || b_is_2d), "Have to provide offsets if there is a 2d matrix, or no offset if both matrices are 3d");

  if (offs.has_value()) {
    TORCH_CHECK(offs->dim() == 1, "offs has to be 1D");
    TORCH_CHECK(offs->dtype() == at::kInt, "Offsets have to be int32");
  }
  TORCH_CHECK(!bias.has_value(), "Bias not supported yet");

  Tensor out = create_grouped_gemm_output_tensor(mat_a, mat_b, offs, out_dtype);

  // at::cuda::detail::bf16bf16_grouped_mm(mat_a, mat_b, offs, bias, out);
  // return out;
  // Forward to the unchanged upstream implementation.
  bf16bf16_grouped_mm(mat_a, mat_b, offs, bias, out);
  return out;
}

TORCH_LIBRARY(groupedmm_ext, m) {
  // Preallocated variant (unchanged)
  // m.def("_grouped_mm_out(Tensor mat_a, Tensor mat_b, Tensor? offs=None, "
  //       "Tensor? bias=None, Tensor out) -> Tensor");
  // New allocator variant
  m.def("_grouped_mm(Tensor mat_a, Tensor mat_b, Tensor? offs=None, "
        "Tensor? bias=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(groupedmm_ext, CUDA, m) {
  // m.impl("_grouped_mm_out", &_grouped_mm_out);
  m.impl("_grouped_mm",     &_grouped_mm);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // No Python bindings here; TORCH_LIBRARY already registers the ops.
}
