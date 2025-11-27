# CUTLASS Deep Dive Documentation

Comprehensive technical documentation for CUTLASS library components, with detailed walkthroughs of data structures, design patterns, and call chains.

## Available Documentation

### Sub-Byte Floating Point Types

Complete documentation of CUTLASS's 4-bit floating point type system, including static types and type-erased (runtime dispatch) variants.

**Location**: [claude/subbyte/](subbyte/)

**Topics Covered**:
- float_e2m1_t (4-bit floating point format)
- nv_float4_t (NVIDIA block-scaling wrapper)
- type_erased_dynamic_float4_t (union-based type erasure)
- Complete inheritance hierarchies and call chains
- Static vs type-erased type comparison
- Design patterns and architectural insights

**Key Files**:
- [README.md](subbyte/README.md) - Table of contents and architecture overview
- [SUMMARY.md](subbyte/SUMMARY.md) - Quick reference summary
- [01-08: Core Type System](subbyte/) - Static types (float_e2m1_t, nv_float4_t)
- [09-11: Type-Erased System](subbyte/) - Runtime dispatch types

### MMA_Atom Design Pattern

Deep dive into the trait-based static polymorphism pattern used for matrix multiply-accumulate operations.

**Location**: [claude/mma-atom-pattern.md](mma-atom-pattern.md)

**Topics Covered**:
- Complete template inheritance chain unpacking
- MMA_Atom → MMA_Traits → Hardware operation
- Trait-based static polymorphism pattern
- Policy-based design
- Friend function injection (Barton-Nackman trick)
- Zero-cost abstraction techniques
- Complete code flow from user code to PTX instructions

**Key Concepts**:
- Template specialization chain
- Compile-time vs runtime dispatch
- ADL (Argument-Dependent Lookup)
- Design pattern comparison and rationale

---

## Documentation Style

All documentation follows a consistent deep-dive style:

1. **Complete Unpacking**: Step-by-step breakdown of inheritance hierarchies
2. **Frame-by-Frame Analysis**: Layer-by-layer explanation at each abstraction level
3. **Annotated Code Snippets**: Relevant code with file locations and line numbers
4. **Visual Diagrams**: ASCII diagrams showing relationships and data flow
5. **Design Pattern Analysis**: Identification and explanation of patterns used
6. **Practical Examples**: Real code traces showing execution flow

---

## Quick Navigation

### For Learning About:

**4-bit Floating Point Types**:
- Start: [subbyte/01-overview.md](subbyte/01-overview.md)
- Quick Reference: [subbyte/SUMMARY.md](subbyte/SUMMARY.md)
- Type Erasure: [subbyte/09-type-erased-float4.md](subbyte/09-type-erased-float4.md)

**MMA Operations and Design Patterns**:
- [mma-atom-pattern.md](mma-atom-pattern.md)

**sizeof_bits Behavior**:
- See: [subbyte/SUMMARY.md](subbyte/SUMMARY.md) - explains why `sizeof_bits_v<nv_float4_t>` returns 8
- Use: `sizeof_bits_v<typename Element::DataType>` for actual bit size

---

## Contributing

When adding new documentation:

1. Follow the deep-dive format (complete unpacking, frame-by-frame analysis)
2. Include code references with file paths and line numbers
3. Use markdown links for clickable navigation: `[file.hpp:42](../include/file.hpp#L42)`
4. Add visual diagrams where helpful
5. Explain the "why" not just the "what"
6. Update this README with links to new documentation

---

## File Organization

```
claude/
├── README.md                  # This file
├── mma-atom-pattern.md        # MMA_Atom design pattern deep dive
└── subbyte/                   # Sub-byte floating point types
    ├── README.md              # Subbyte documentation index
    ├── SUMMARY.md             # Quick reference
    ├── 01-overview.md         # Introduction to float_e2m1_t
    ├── 02-type-hierarchy.md   # Complete inheritance chain
    ├── 03-fpbitrepresentation.md  # Bit-level foundation
    ├── 04-float-exmy-base.md  # CRTP base class
    ├── 05-float-e2m1.md       # Concrete E2M1 type
    ├── 06-nv-float4.md        # NVIDIA scaling wrapper
    ├── 07-call-chains.md      # Step-by-step examples
    ├── 08-design-patterns.md  # Architecture insights
    ├── 09-type-erased-float4.md       # Union-based type erasure
    ├── 10-type-erased-call-chains.md  # Runtime format selection
    └── 11-type-erased-diagrams.md     # Visual architecture
```

---

**Last Updated**: 2025-11-16
