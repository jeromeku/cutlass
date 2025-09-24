/*
clang -cc1 -fdump-record-layouts crtp.cpp 
clang -cc1 -fdump-vtable-layouts crtp.cpp 

*/
// #include <iostream>
struct B { virtual ~B() = default; virtual int f() const { return 1; } int x; };
struct D : B { int y; int f() const override { return x + y; } };

struct S { int x; };               // no virtuals
struct T : S { int y; };           // still no virtuals

// int main() {
//     std::cout << "sizeof(B) = " << sizeof(B) << "\n"; // includes vptr + int
//     std::cout << "sizeof(D) = " << sizeof(D) << "\n"; // vptr + two ints (and padding)
//     std::cout << "sizeof(S) = " << sizeof(S) << "\n";
//     std::cout << "sizeof(T) = " << sizeof(T) << "\n";
// }
