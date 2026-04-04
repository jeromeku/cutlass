#include "cute/layout.hpp"
#include "cute/stride.hpp"
#include "cute/tensor.hpp"

using namespace cute;

int main(){

    using Atom = Layout<Shape<Int<32>, Int<8>>, Stride<Int<8>, Int<1>>>;
    using Tiler = Shape<_128, _64>;  // Include 7 pipeline stages
    auto tiled_by_row = tile_to_shape(Atom{}, Tiler{}, GenRowMajor{});
    auto tiled_by_col = tile_to_shape(Atom{}, Tiler{}, GenColMajor{});
    print(tiled_by_row);
    printf("\n");
    print(tiled_by_col);
}