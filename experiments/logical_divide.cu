#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print_tensor.hpp"

using namespace cute;

int main() {
  auto shape = Shape<_9, Shape<_4, _8>>{};
  auto stride = make_stride(Int<59>{}, make_stride(Int<13>{}, Int<1>{}));
  auto layout = make_layout(shape, stride);
  auto tiler =
      make_tile(Layout<_3, _3>{}, Layout<Shape<_2, _4>, Stride<_1, _8>>{});
  auto result = logical_divide(layout, tiler);
  printf("Tiler\n");
  print(tiler);
  printf("\n");
  printf("Original layout:\n");
  print_layout(layout);
  printf("\n");
  printf("Logical divided layout:\n");
  print_layout(result);
}
