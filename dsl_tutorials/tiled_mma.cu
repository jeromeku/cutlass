#include "cute/tensor.hpp"
#include <vector>
using namespace cute;

#define PRINT_CUTE(label, cute_obj) \
    printf("%s: ", label); cute::print(cute_obj); printf("\n"); \

#define PRINT_CUTE_TENSOR(label, cute_obj) \
    printf("%s: ", label); cute::print_tensor(cute_obj); printf("\n"); \

int main(){
  constexpr int M = 32;
  constexpr int N = 32;
  constexpr int K = 16;
  using T = int;
  std::vector<T> A(M * K);
  std::vector<T> B(N * K);
  std::vector<T> C(M * N);
  for(int i = 0; i < A.size(); i++){
    A[i] = i;
  }

  for(int i = 0; i < B.size(); i++){
    B[i] = i;
  }

  for(int i = 0; i < C.size();i++){
    C[i] = i;
  }
  
  auto gA = make_tensor(A.data(), Layout<Shape<Int<M>, Int<K>>, Stride<Int<K>, Int<1>>>{});
  auto gB = make_tensor(B.data(), Layout<Shape<Int<N>, Int<K>>, Stride<Int<1>, Int<N>>>{});
  auto gC = make_tensor(C.data(), Layout<Shape<Int<M>, Int<N>>, Stride<Int<N>, Int<1>>>{});
  
  using MMA_op = SM80_16x8x8_F32BF16BF16F32_TN;
  using MMA_traits = MMA_Traits<MMA_op>;
  using MMA_atom = MMA_Atom<MMA_traits>;
  using MMA_shape = MMA_traits::Shape_MNK;

  static constexpr int kMmaThrExpandM = 2;
  static constexpr int kMmaThrExpandN = 4;
  static constexpr int kMmaThrExpandK = 1;

  static constexpr int kMmaValExpandM = 1;
  static constexpr int kMmaValExpandN = 1;
  static constexpr int kMmaValExpandK = 2;

  static constexpr int kMmaTileM = kMmaThrExpandM * kMmaValExpandM * get<0>(MMA_shape{});
  static constexpr int kMmaTileN = kMmaThrExpandN * kMmaValExpandN * get<1>(MMA_shape{});
  static constexpr int kMmaTileK = kMmaThrExpandK * kMmaValExpandK * get<2>(MMA_shape{});

  using MMAThrLayout = decltype(make_layout(make_shape(Int<kMmaThrExpandM>{},
                                                       Int<kMmaThrExpandN>{},
                                                       Int<kMmaThrExpandK>{})));
  using MMATileLayout = Tile<Int<kMmaTileM>, Int<kMmaTileN>, Int<kMmaTileK>>;
  using TiledMMA = decltype(make_tiled_mma(MMA_op{}, MMAThrLayout{}, MMATileLayout{}));
  auto tiled_mma = TiledMMA{};
//   printf("Tiled mma: "); print(tiled_mma); printf("\n");
 
  auto thr_mma = tiled_mma.get_slice(0);
  auto tCgA = thr_mma.partition_A(gA);
  auto tCgB = thr_mma.partition_B(gB);
  auto tCgC = thr_mma.partition_C(gC);

#if defined(PRINT_LATEX)
  print_latex(tiled_mma);
#else
//   PRINT_CUTE("tCgA", tCgA);
//   PRINT_CUTE("tCgB", tCgB);
//   PRINT_CUTE("tCgC", tCgC);
  PRINT_CUTE_TENSOR("tCgA", tCgA); 
  PRINT_CUTE_TENSOR("tCgB", tCgB); 
  PRINT_CUTE_TENSOR("tCgC", tCgC); 

  #endif
}