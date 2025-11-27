#include "cute/layout.hpp"
#include "cute/pointer.hpp"
#include "tma_load_testbed.hpp"
#include <thrust/host_vector.h>

using namespace cute;
using namespace cutlass::test;

template <class T, class TmaType = T, class GMEM_Layout, class SMEM_Layout,
          class CTA_Tile>
auto test_tma_load(GMEM_Layout const &gmem_layout,
                   SMEM_Layout const &smem_layout, CTA_Tile const &cta_tile) {
  return test_tma_load<T, TmaType>(SM90_TMA_LOAD{}, gmem_layout, smem_layout,
                                   cta_tile);
}

template <class T, class TmaType = T, class GMEM_Layout, class SMEM_Layout>
auto test_tma_load(GMEM_Layout const &gmem_layout,
                   SMEM_Layout const &smem_layout) {
  return test_tma_load<T, TmaType>(gmem_layout, smem_layout,
                                   product_each(shape(smem_layout)));
}

template <class Fn> void print_matrix(int cols, int rows, Fn &&fn) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      auto coord = make_coord(i, j);
      const auto val = std::invoke(fn, coord);
      printf("%-4u", val);
    }
    printf("\n");
  }
}
template <typename T, class Fn> 
void print_swizzle(int cols, int rows, Fn &&fn) {
  auto bytes = sizeof(T);
  const int effective_cols = bytes * cols;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      const int effective_col = bytes * j;
      const int coord = i * effective_cols + effective_col;
      const auto val = fn.apply(coord);
      printf("%-4u", static_cast<T>(val / bytes));
    }
    printf("\n");
  }
}
int main() {
  constexpr int N = 64;

  using T = uint16_t;
  constexpr int B = 1;
  // constexpr int N = sizeof_bits<T>::value;

  auto swizzle_atom = GMMA::Layout_K_SW128_Atom<T>{};
  constexpr int atom_cols = size<1>(swizzle_atom.layout_b());
  constexpr int atom_rows = size<0>(swizzle_atom.layout_b());
  auto layout_a = swizzle_atom.layout_a();
  auto layout_b = swizzle_atom.layout_b();
  constexpr int numel = atom_cols * atom_rows;
  thrust::host_vector<T> mat(numel);
  for(int i = 0; i < numel; i ++){
    mat[i] = i % atom_cols;
  }
  print_swizzle<T>(64, 4, layout_a);
  
  // auto composed_layout = ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag, Layout<Shape<_8,_1024>,Stride<_1024,_1>>>{}; 
  // // auto layout = upcast<N>(composed_layout); auto layout_b = composed_layout.layout_b();
  // // auto upcasted_layout_b = upcast<N>(layout_b.shape(), layout_b.stride());
  // auto _offset = smem_ptr_flag_bits<B * N>{};
  // // auto composed = composition(composed_layout.layout_a(),
  // // smem_ptr_flag_bits<B*N>{}, upcasted_layout_b); cute::print(composed);
  // // printf("\n");
  // cute::print(swizzle_atom);
  // printf("\n");
  // auto offset = swizzle_atom.offset();
  // print(offset);
  // print("\n");
  
  // // printf("No offset:\n");
  // // print_matrix(atom_cols, 4,
  // //              [&](auto coord) { return layout_a(layout_b(coord)); });
  // printf("Swizzle atom:\n");
  // print_matrix(atom_cols, 4, swizzle_atom);
  // printf("\n");
  // Layout smem_layout = Layout<Shape<Int<N>,Int<N>>, Stride<Int<N>,_1>>{};
  // constexpr auto swizzled_layout = tile_to_shape(swizzle_atom, smem_layout);
  // Layout gmem_layout = smem_layout;
  // test_tma_load<T>(gmem_layout, swizzled_layout);
}
