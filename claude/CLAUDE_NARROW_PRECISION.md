Help me understand the Blackwell narrow precision gemm:

Start by tracing through the following example:
/home/jeromeku/cutlass/examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu

**Important**: Cutlass code is heavily templated. Please make sure to fully trace through the entire dispatched path for each templated type / function.  I want to understand the specialized call paths **end-to-end** from user call -> fully unrolled call stack that peals away each layer of abstraction.

**Remember**: The idea is to provide **literate code**: a guided **line-by-line** walkthrough of the execution flow with **annotated inline code** snippets and **mappings** to corresponding code sections (source file + line number spans).  

*Think pedagogic narrative + code + clickable links + visuals that guide the user, frame by frame, through the entire call graph -- a detailed guided tour of the system architecture with pointers for additional exploration.*

As a supplement:
1) Show how each of the above host / device templated objects / functions can be tested in isolation:
- provide minimal code snippets that can be run in isolation 
- goal of these minimal examples is to be able to poke into the internals and experiment with these APIs separate from the complexities of running examples end-to-end.

2) Document where else the narrow precision utilities / types used in the narrow precision example are tested
- ideally I'd like to test each of these components in a unittest-like setting

Liberal use of markdown visuals (tables, diagrams, etc.) is encouraged.

Code links should be clickable from within the VScode editor (when viewed as a markdown doc): e.g.,
[barrier.cuh:196](src/include/non_abi/device/coll/barrier.cuh#L196)

Do all your work in a folder "claude/blackwell-narrow-precision-gemm".