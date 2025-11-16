# Complete Call Chains: Step-by-Step Examples

This chapter provides **complete, annotated traces** of common operations with `float_e2m1_t`.

## Example 1: Construction from Float

### Code

```cpp
float_e2m1_t x(1.5f);
```

### Complete Call Chain

```
1. User Code
   float_e2m1_t x(1.5f);
   │
   ▼
2. float_e2m1_t Constructor
   [float_subbyte.h:90]
   explicit float_e2m1_t(float x) : Base(x) {}
   │
   │ Calls base class constructor with 1.5f
   ▼
3. float_exmy_base<E2M1, float_e2m1_t> Constructor
   [exmy_base.h:1024]
   explicit float_exmy_base(float x) {
       storage = static_cast<Derived*>(this)->convert_from_float(x).storage;
   }
   │
   │ CRTP cast: static_cast<float_e2m1_t*>(this)
   │ Calls: float_e2m1_t::convert_from_float(1.5f)
   │ (float_e2m1_t doesn't override, so uses base class default)
   ▼
4. float_exmy_base::convert_from_float
   [exmy_base.h:1004]
   float_exmy_base convert_from_float(float const &flt) const {
       FP32BitRepresentation::Storage fp32_bits =
           FP32BitRepresentation::to_bits(flt);
   }
   │
   │ Extract raw bits from 1.5f
   ▼
5. FpBitRepresentation<uint32_t, 32, 8, 23, ...>::to_bits
   [exmy_base.h:562]
   template <class FpType>
   static Storage to_bits(FpType flt) {
       return copy_bits<FpType, Storage>(flt);
   }
   │
   │ Uses memcpy to reinterpret 1.5f as uint32_t
   │ Result: 0x3FC00000
   │
   │ IEEE-754 breakdown:
   │ 0x3FC00000 = 00111111110000000000000000000000
   │              S│Exponent│Mantissa
   │              0│01111111│10000000000000000000000
   │              +│  127   │ 0.5
   │              = (+1) × 1.5 × 2^(127-127) = 1.5
   ▼
6. Back to convert_from_float
   float_exmy.storage = BitRepresentation::convert_from(
       fp32_bits,                    // 0x3FC00000
       FP32BitRepresentation{}       // FP32 encoder object
   );
   │
   ▼
7. FpBitRepresentation<uint8_t, 4, 2, 1, ...>::convert_from
   [exmy_base.h:575]
   static Storage convert_from(
       typename SrcFpBits::Storage src_val,
       SrcFpBits src_encoding) {
       return convert(src_encoding, src_val, FpBitRepresentation{});
   }
   │
   │ Calls the master convert() function
   ▼
8. FpBitRepresentation::convert
   [exmy_base.h:678-845]
   template <class SrcFpBits, class DstFpBits>
   static typename DstFpBits::Storage convert(
       SrcFpBits src_encoding,               // FP32 encoding
       typename SrcFpBits::Storage src_val,  // 0x3FC00000
       DstFpBits dst_encoding)               // E2M1 encoding

   Step 8a: Extract source components
   ┌─────────────────────────────────────────────────┐
   │ src_sign_bit = src_encoding.sign_bit(0x3FC00000)│
   │              = 0x3FC00000 >> 31                  │
   │              = 0 (positive)                      │
   ├─────────────────────────────────────────────────┤
   │ src_exp_bits = src_encoding.exponent_bits(...)  │
   │              = (0x3FC00000 >> 23) & 0xFF        │
   │              = 0x7F = 127                        │
   ├─────────────────────────────────────────────────┤
   │ src_significand = src_encoding.significand(...) │
   │                 = (1 << 23) | mantissa_bits     │
   │                 = 0x800000 | 0x400000           │
   │                 = 0xC00000                       │
   │                 = 1.1₂ (binary: 1.5 decimal)    │
   ├─────────────────────────────────────────────────┤
   │ src_exp = src_encoding.exponent(...)            │
   │         = 127 - 127 = 0                         │
   └─────────────────────────────────────────────────┘

   Step 8b: Check for special values
   ┌─────────────────────────────────────────────────┐
   │ if (src_encoding.is_nan(0x3FC00000))            │
   │     return dst_encoding.nan_with_sign(...)      │
   │ → false, 1.5 is not NaN                         │
   ├─────────────────────────────────────────────────┤
   │ if (src_encoding.is_inf(0x3FC00000))            │
   │     return dst_encoding.inf_with_sign(...)      │
   │ → false, 1.5 is not Inf                         │
   ├─────────────────────────────────────────────────┤
   │ if (src_exp_bits == 0 && src_significand == 0)  │
   │     return dst_encoding.set_sign_bit(0, 0)      │
   │ → false, 1.5 is not zero                        │
   └─────────────────────────────────────────────────┘

   Step 8c: Normalize (if denormal)
   ┌─────────────────────────────────────────────────┐
   │ while (significand_hidden_bits(src_significand) │
   │        == 0) {                                   │
   │     src_significand <<= 1;                       │
   │     src_exp--;                                   │
   │ }                                                │
   │ → Loop doesn't execute (1.5 is normal)          │
   │   Hidden bit already set: 1.1₂                  │
   └─────────────────────────────────────────────────┘

   Step 8d: Check destination exponent range
   ┌─────────────────────────────────────────────────┐
   │ DstFpBits::MAX_EXP = 2                          │
   │ DstFpBits::MIN_EXP = 0                          │
   │                                                  │
   │ if (src_exp > MAX_EXP)  // 0 > 2? No            │
   │ if (src_exp >= MIN_EXP && src_exp <= MAX_EXP)   │
   │     // 0 >= 0 && 0 <= 2? Yes!                   │
   │     → Exponent fits, proceed normally           │
   └─────────────────────────────────────────────────┘

   Step 8e: Convert mantissa
   ┌─────────────────────────────────────────────────┐
   │ shift_amount = DstBits::NUM_MANTISSA_BITS       │
   │              - SrcBits::NUM_MANTISSA_BITS       │
   │              = 1 - 23 = -22                     │
   │ (Negative: right shift to reduce precision)     │
   ├─────────────────────────────────────────────────┤
   │ dst_exponent = src_exp + DstBits::EXP_BIAS      │
   │              = 0 + 1 = 1                        │
   ├─────────────────────────────────────────────────┤
   │ dst_mantissa = src_significand                  │
   │              = 0xC00000                          │
   │              = 110000000000000000000000₂        │
   │              = 1.1₂ × 2^23                      │
   │                                                  │
   │ dst_mantissa = round_significand(dst_mantissa,  │
   │                                   -22)          │
   └─────────────────────────────────────────────────┘

   Step 8f: Round to nearest even
   ┌─────────────────────────────────────────────────┐
   │ round_significand(0xC00000, -22):               │
   │                                                  │
   │ Shift right by 22 bits:                         │
   │ 110000000000000000000000₂ >> 22                 │
   │ = 11₂ = 3                                       │
   │                                                  │
   │ Check rounding:                                  │
   │ Guard bit (bit 22) = 0                          │
   │ Round bit (bit 21) = 0                          │
   │ Sticky bits (bits 0-20) = 0                     │
   │                                                  │
   │ No rounding needed (all zero)                   │
   │ Result: 3 = 0b11                                │
   │                                                  │
   │ Extract mantissa bits (1 bit):                  │
   │ dst_mantissa &= MANTISSA_MASK                   │
   │               = 3 & 0b1 = 0b1                   │
   └─────────────────────────────────────────────────┘

   Step 8g: Assemble final value
   ┌─────────────────────────────────────────────────┐
   │ make_fp_from_bits(sign, exp, mantissa):         │
   │                                                  │
   │ fp_bits = (sign << SIGN_SHIFT)                  │
   │         | (exp << EXPONENT_SHIFT)               │
   │         | mantissa                              │
   │                                                  │
   │         = (0 << 3) | (1 << 1) | 1               │
   │         = 0b0000 | 0b0010 | 0b0001              │
   │         = 0b0011                                 │
   │                                                  │
   │ Verification:                                    │
   │ 0b0011 = S:0 E:01 M:1                           │
   │        = (+1) × 1.1₂ × 2^(1-1)                  │
   │        = 1 × 1.5 × 2^0                          │
   │        = 1.5 ✓                                  │
   │                                                  │
   │ Return: 0b0011                                   │
   └─────────────────────────────────────────────────┘
   │
   ▼
9. Back to convert_from_float
   float_exmy.storage = 0b0011;
   return float_exmy;
   │
   ▼
10. Back to float_exmy_base constructor
    storage = float_exmy.storage;
    storage = 0b0011;
    │
    ▼
11. Base class constructor returns
    float_e2m1_t object constructed
    with storage = 0b0011
    │
    ▼
12. Result
    float_e2m1_t x;
    x.storage = 0b0011 = 1.5

    Memory:
    ┌───────────┐
    │ uint8_t   │
    │ 00000011  │
    └───────────┘
```

### Summary

```
Input:  1.5f (32-bit float)
↓
IEEE-754: 0x3FC00000 = 0 01111111 10000000000000000000000
↓
Extract:  sign=0, exp=0, significand=1.1₂
↓
Convert:  exp=0→1 (bias adjustment)
          mantissa: 1.1₂ → 1₂ (truncate to 1 bit, no rounding needed)
↓
Assemble: 0b0011 = S:0 E:01 M:1
↓
Output:   0b0011 stored in uint8_t
Verification: 0b0011 = 1.5 ✓
```

---

## Example 2: Arithmetic Operation

### Code

```cpp
float_e2m1_t a(1.0f);
float_e2m1_t b(0.5f);
float_e2m1_t c = a + b;
```

### Complete Call Chain

```
1. Construction of a
   float_e2m1_t a(1.0f);
   [Similar to Example 1]
   Result: a.storage = 0b0010
   ┌─────────────────┐
   │ 0b0010          │
   │ = S:0 E:01 M:0  │
   │ = 1.0 × 2^0     │
   │ = 1.0           │
   └─────────────────┘

2. Construction of b
   float_e2m1_t b(0.5f);
   [Similar to Example 1]
   Result: b.storage = 0b0001
   ┌─────────────────┐
   │ 0b0001          │
   │ = S:0 E:00 M:1  │
   │ = 0.1 × 2^0     │
   │ = 0.5 (denorm)  │
   └─────────────────┘

3. Addition operator+
   [exmy_base.h:1129]
   friend float_exmy_base operator+(
       float_exmy_base const &lhs,
       float_exmy_base const &rhs) {
       return float_exmy_base(float(lhs) + float(rhs));
   }
   │
   │ Step 3a: Convert lhs to float
   │ Step 3b: Convert rhs to float
   │ Step 3c: FP32 addition
   │ Step 3d: Convert result back to E2M1
   ▼

4. Convert lhs to float
   float(a)
   │
   ├─ Implicit conversion operator
   │  [exmy_base.h:1041]
   │  operator float() const {
   │      return static_cast<const Derived*>(this)->convert_to_float(*this);
   │  }
   │
   ├─ convert_to_float
   │  [exmy_base.h:1012]
   │  float convert_to_float(float_exmy_base const &x) const {
   │      FP32BitRepresentation::Storage fp32_bits;
   │      fp32_bits = BitRepresentation::convert_to(
   │          x.storage,                    // 0b0010
   │          FP32BitRepresentation{}       // FP32 target
   │      );
   │      return copy_bits<uint32_t, float>(fp32_bits);
   │  }
   │
   ├─ convert_to (E2M1 → FP32)
   │  Input: 0b0010 (E2M1 for 1.0)
   │
   │  Extract E2M1 components:
   │  ┌─────────────────────────────┐
   │  │ sign = 0                    │
   │  │ exp_bits = 01 = 1           │
   │  │ exp = 1 - 1 = 0             │
   │  │ mantissa = 0                │
   │  │ significand = 1.0₂          │
   │  └─────────────────────────────┘
   │
   │  Convert to FP32:
   │  ┌─────────────────────────────┐
   │  │ dst_exp = 0 + 127 = 127     │
   │  │ dst_mantissa = 0.0 << 22    │
   │  │              = 0x000000     │
   │  │                              │
   │  │ fp32_bits = (0 << 31)       │
   │  │           | (127 << 23)     │
   │  │           | 0x000000        │
   │  │           = 0x3F800000      │
   │  └─────────────────────────────┘
   │
   └─ Result: 1.0f

5. Convert rhs to float
   float(b)
   [Similar process]

   Input: 0b0001 (E2M1 for 0.5)
   Extract: sign=0, exp_bits=00 (denormal), mantissa=1
   Denormal: significand = 0.1₂ = 0.5, exp = 0

   Convert to FP32:
   ┌─────────────────────────────┐
   │ dst_exp = 0 + 127 = 127     │
   │ dst_mantissa = 0.0          │
   │ Wait, denormal needs special handling!
   │                              │
   │ For denormal 0.5:           │
   │ Real value = 0.1₂ × 2^0     │
   │ In FP32: 2^-1 = 0.5         │
   │ exp_bits = 126              │
   │ mantissa = 0                │
   │ fp32_bits = 0x3F000000      │
   └─────────────────────────────┘

   Result: 0.5f

6. FP32 Addition
   1.0f + 0.5f = 1.5f
   │
   │ IEEE-754 FP32 addition (hardware)
   │ Result: 0x3FC00000 = 1.5f
   ▼

7. Convert result back to E2M1
   float_exmy_base(1.5f)
   │
   │ [Same as Example 1, Step 2-10]
   │
   └─ Result: 0b0011

8. Final result
   float_e2m1_t c;
   c.storage = 0b0011 = 1.5

   ┌─────────────────┐
   │ 0b0011          │
   │ = S:0 E:01 M:1  │
   │ = 1.1₂ × 2^0    │
   │ = 1.5           │
   └─────────────────┘
```

### Summary

```
a (1.0)         b (0.5)
0b0010          0b0001
   ↓               ↓
convert_to_float  convert_to_float
   ↓               ↓
1.0f (FP32)     0.5f (FP32)
   └──────┬────────┘
          ↓
    FP32 addition
    (1.0f + 0.5f)
          ↓
       1.5f
          ↓
  convert_from_float
          ↓
       0b0011
          ↓
    c (1.5)
```

---

## Example 3: Type Usage in GEMM

### Code

```cpp
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
typename ElementA::DataType matrix_value(2.0f);
typename ElementA::ScaleFactorType scale_value(4.0f);
float real_value = float(matrix_value) * float(scale_value);
```

### Call Chain

```
1. Type Alias
   using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;

   Expands to:
   ┌────────────────────────────────────────┐
   │ struct ElementA {                      │
   │     using DataType = float_e2m1_t;     │
   │     using ScaleFactorType =            │
   │         float_ue4m3_t;                 │
   │ };                                     │
   └────────────────────────────────────────┘

2. Construct matrix_value (E2M1)
   typename ElementA::DataType matrix_value(2.0f);
   = float_e2m1_t matrix_value(2.0f);

   [Same as Example 1]
   Result: matrix_value.storage = 0b0100
   ┌─────────────────┐
   │ 0b0100          │
   │ = S:0 E:10 M:0  │
   │ = 1.0 × 2^1     │
   │ = 2.0           │
   └─────────────────┘

3. Construct scale_value (UE4M3)
   typename ElementA::ScaleFactorType scale_value(4.0f);
   = float_ue4m3_t scale_value(4.0f);

   float_ue4m3_t is 8-bit: UE4M3 format
   ┌────────────────────┐
   │ Bit: 7 6 5 4 3 2 1 0│
   │     ┌───────┬──────┐│
   │     │ E E E E│M M M││
   │     └───────┴──────┘│
   │     4 exp   3 mant  │
   └────────────────────┘

   4.0 = 1.0 × 2^2
   ┌────────────────────┐
   │ Exponent = 2       │
   │ Bias = 7           │
   │ Biased exp = 2+7=9 │
   │ Exp bits = 1001    │
   │ Mantissa = 0       │
   │ Bits: 10010000     │
   │     = 0x90         │
   └────────────────────┘

   Result: scale_value.storage = 0x90

4. Convert matrix_value to float
   float(matrix_value)
   = 2.0f

5. Convert scale_value to float
   float(scale_value)
   = 4.0f

6. FP32 Multiplication
   2.0f * 4.0f = 8.0f

7. Result
   real_value = 8.0f
```

### Memory Layout

```
In actual GEMM operation:
┌────────────────────────────────────────────┐
│ Matrix A (scaled):                         │
│                                             │
│ Block 0 (32 elements):                     │
│ ┌─────────┬──────────────────────────────┐ │
│ │ Scale   │ Data                         │ │
│ │ 0x90    │ 0b0100 0b0011 0b0110 ...    │ │
│ │ (4.0)   │ (2.0)  (1.5)  (4.0)  ...    │ │
│ └─────────┴──────────────────────────────┘ │
│   1 byte    32 × 4 bits = 16 bytes         │
│                                             │
│ Real values: [8.0, 6.0, 16.0, ...]         │
│ (data × scale for each)                    │
└────────────────────────────────────────────┘

Total per block: 17 bytes
Compression vs FP16: (2*32) / 17 = 3.76x
```

---

## Key Takeaways

1. **All arithmetic goes through float**: E2M1 has no native arithmetic, always converts to FP32

2. **Conversion is IEEE-754-like**: Uses standard rounding, exponent adjustment, mantissa truncation

3. **CRTP enables customization**: Each type can override conversion methods

4. **Zero overhead**: All conversions compile to simple bit manipulation

5. **Type traits enable metaprogramming**: `nv_float4_t` packages types for template dispatch

---

**Next**: [08-design-patterns.md](08-design-patterns.md) - Design patterns and architectural insights
