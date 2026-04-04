#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print_tensor.hpp"
#include "debug.hpp"

using namespace cute;

int main(){
    auto layout = Layout<Shape<_2,_2>, Stride<_1,_6>>{};
    printf("Layout:\n");
    print_layout(layout);
    auto complement = cute::complement(layout, 24);
    printf("Complement:\n");
    print_layout(complement);
    printf("\n");
    auto concatenated = make_layout(layout, complement);
    printf("Concatenated:\n");
    print_layout(concatenated);
    printf("\n");
    
}


