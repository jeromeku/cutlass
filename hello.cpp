/*
clang -### hello.cpp
clang -ccc-print-phases hello.cpp
clang -O3 -Rpass=.* -Rpass-missed=.* -Rpass-analysis=.*
clang -O0 -S -emit-llvm foo.cc -o foo.ll
opt -passes='default<O2>' -S foo.ll -o -        # new pass manager

*/

#include<stdio.h>

int main(){
    printf("Hello World\n");
}