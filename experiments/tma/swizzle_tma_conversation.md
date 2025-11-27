# CuTe / TMA / Swizzle – Conversation Notes

## Topics Covered
- `printf` left alignment and field width.
- Templating `print_matrix` over a callable.
- Passing `GMMA::Layout_K_SW128_Atom` and composed layouts as callables.
- Why some `print_matrix` calls produced identical output.
- How `GMMA::Layout_K_SW128_Atom` is defined and expanded.
- Difference between host “swizzle atom” prints and TMA‑loaded SMEM prints.
- How “unswizzling” works when reading from swizzled SMEM.
- How `Swizzle<3,4,3>` operates on byte addresses.
- How the element type (e.g. `uint16_t`) is reconciled with a bit‑based swizzle.
- GDB/LLDB breakpoint options on CuTe layouts.

## `printf` Left Alignment
- Left‑align with width 4 using the `-` flag, e.g. `printf("%-4u", val);`.

## Templated `print_matrix`
- Idiomatic signature:
  ```cpp
  template <class Fn>
  void print_matrix(int cols, int rows, Fn&& fn) {
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        auto coord = make_coord(i, j);
        const auto val = std::invoke(fn, coord);
        printf("%-4u", val);
      }
      printf("\n");
    }
  }
  ```
- `Fn&&` is a forwarding reference, so you can pass lambdas, function objects, or lvalues like `swizzle_atom` with zero overhead.

## Passing `swizzle_atom` and Layout Compositions
- `swizzle_atom` is a `GMMA::Layout_K_SW128_Atom<T>` object with `operator()(Coord)` defined via `ComposedLayout`.
- You can call:
  ```cpp
  print_matrix(atom_cols, 4,
               [&](auto coord){ return layout_a(layout_b(coord)); });
  print_matrix(atom_cols, 4, swizzle_atom);
  ```
- Both calls are equivalent here because `swizzle_atom(coord)` is implemented as `layout_a(offset() + layout_b(coord))` and `offset()` is the zero placeholder.

## `ComposedLayout` and `Layout_K_SW128_Atom`
- In `include/cute/layout_composed.hpp`, a `ComposedLayout<LayoutA, Offset, LayoutB>` represents `LayoutA o Offset o LayoutB`:
  ```cpp
  template <class Coord>
  auto operator()(Coord const& coord) const {
    if constexpr (has_underscore<Coord>::value) {
      return slice(coord, *this);
    } else {
      return layout_a()(offset() + layout_b()(coord)); // (A o O o B)(c)
    }
  }
  ```
- In `include/cute/atom/mma_traits_sm90_gmma.hpp`, the bit‑level K‑major atom is:
  ```cpp
  using Layout_K_SW128_Atom_Bits =
    ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag,
                   Layout<Shape<_8,_1024>,Stride<_1024,_1>>>;

  template <class Type>
  using Layout_K_SW128_Atom =
    decltype(upcast<sizeof_bits<Type>::value>(Layout_K_SW128_Atom_Bits{}));
  ```
- `smem_ptr_flag` / `smem_ptr_flag_bits<B>` is an `Int<0>` placeholder whose template parameter tracks bit width, not a nonzero offset.

## `upcast` and `smem_ptr_flag_bits`
- `upcast<sizeof_bits<Type>::value>` converts a bit‑level layout to an element‑level layout:
  ```cpp
  template <int N, class SwizzleFn, int B, class Layout>
  auto upcast(ComposedLayout<SwizzleFn, smem_ptr_flag_bits<B>, Layout> const& layout) {
    return composition(layout.layout_a(),
                       smem_ptr_flag_bits<B*N>{},
                       upcast<N>(layout.layout_b()));
  }
  ```
- For `uint16_t`, `B` goes from `1` to `16`, but the *value* remains zero (it’s still an `Int<0>`).
- `make_tensor` for a `ComposedLayout<SwizzleFn, smem_ptr_flag_bits<B>, Layout>` enforces that `B` matches the iterator element bit width and wraps the pointer in a swizzle pointer:
  ```cpp
  template <class Iterator, class SwizzleFn, int B, class Layout>
  auto make_tensor(Iterator const& ptr,
                   ComposedLayout<SwizzleFn, smem_ptr_flag_bits<B>, Layout> const& layout)
  {
    static_assert(is_smem<Iterator>::value, "Expected smem.");
    static_assert(B == sizeof_bits<iter_value_t<Iterator>>::value,
                  "Expected a B-bit pointer type.");
    return make_tensor(make_smem_ptr(ptr.get(), layout.layout_a()),
                       layout.layout_b());
  }
  ```

## Swizzle Pointer and Byte‑Address Swizzle
- `swizzle_ptr` wraps an iterator and applies the swizzle to the *byte address*:
  ```cpp
  template <class SwizzleFn, class Iterator>
  struct swizzle_ptr : iter_adaptor<Iterator, swizzle_ptr<SwizzleFn,Iterator>> {
    template <class T>
    static T* apply_swizzle(T* ptr) {
      return reinterpret_cast<T*>(SwizzleFn::apply(reinterpret_cast<uintptr_t>(ptr)));
    }

    reference operator*() const {
      return *apply_swizzle(this->get());
    }

    template <class Int>
    reference operator[](Int const& i) const {
      return *apply_swizzle(this->get() + i);
    }
  };
  ```
- `this->get() + i` is pointer arithmetic in units of `sizeof(T)`. For `uint16_t`, the effective byte address is `base + i * 2`.
- `Swizzle<3,4,3>::apply` then xors bits of this byte address:
  ```cpp
  template <int BBits, int MBase, int SShift>
  struct Swizzle {
    using bit_msk = cute::constant<int, (1 << num_bits) - 1>;
    using yyy_msk = cute::constant<int, bit_msk{} << (num_base + max(0,num_shft))>;
    using zzz_msk = cute::constant<int, bit_msk{} << (num_base - min(0,num_shft))>;

    template <class Offset>
    static auto apply(Offset const& offset) {
      return offset ^ shiftr(offset & yyy_msk{}, msk_sft{}); // ZZZ ^= YYY
    }
  };
  ```

## Host “Swizzle Atom” vs TMA SMEM Prints
- Host “Swizzle atom” print:
  - Uses `swizzle_atom` directly on logical coords: `swizzle_atom(coord)`.
  - Offset is the zero placeholder, so it shows a single 8×64 atom’s mapping with base address 0.
- TMA SMEM prints:
  - TMA is configured with `swizzled_layout = tile_to_shape(swizzle_atom, smem_layout)` and writes into SMEM using that layout.
  - `sA_unswizzled` uses a plain row‑major `gmem_layout` on the same SMEM buffer, so it reveals the physical swizzle pattern when printed.
  - `sA` uses the swizzled `smem_layout`; because the tensor view matches the write layout, reading through it effectively “unswizzles” and yields the original logical order initialized on the host.

## Why “Raw Swizzle on Offsets” Looks Different
- The swizzle always acts on *byte addresses*.
- Actual SMEM access:
  1. Layout computes an element index from `(row, col)`.
  2. Pointer arithmetic converts element index to byte address: `addr = base + elem * sizeof(T)`.
  3. `Swizzle<3,4,3>::apply` permutes bits of `addr`.
  4. Result is reinterpreted as `T*` and dereferenced.
- If you instead apply `Swizzle<3,4,3>` directly to the element index as if it were a byte address, you implicitly assume `sizeof(T)==1`, which changes which bits correspond to element vs. byte positions and yields a different “chunk” and period pattern.

## GDB/LLDB Breakpoints on `ComposedLayout::operator()`
- Line break:
  - GDB: `break include/cute/layout_composed.hpp:114`
  - LLDB: `break set --file include/cute/layout_composed.hpp --line 114`
- Regex or symbol‑based breakpoints can be used to target specific instantiations of `operator()(Coord const&)`.

