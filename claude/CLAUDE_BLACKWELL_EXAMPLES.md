Help me understand the following examples:
/home/jeromeku/cutlass/examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu
/home/jeromeku/cutlass/examples/cute/tutorial/blackwell/04_mma_tma_2sm_sm100.cu
/home/jeromeku/cutlass/examples/cute/tutorial/blackwell/05_mma_tma_epi_sm100.cu

Note that there is significant overlap in these examples -- explain the common parts once and then create specific sections for the unique parts.

Specifically:
- Focus on the host setup (`gemm_host_f16xf16_f32_f32_tnt`) and `gemm_device` kernel
- Pay particular attention to each templated call:
    - For host code:
        - `make_tiled_mma`
        - `UMMA::tile_to_mma_shape`
        - `make_tma_atom`
        - `tma_atom_A.get_tma_tensor`
    - For device code:
        - `tma_partition`
        - `create_tma_multicast_mask`
        - `copy`
        - `gemm`
        - `make_tmem_copy`
        - `thr_t2r_copy.partition_S`
        - `tiled_t2r_copy.get_slice`
        - `tmem_allocator.allocate`
        - `cutlass::arch::umma_arrive_multicast_2x1SM`

**Important**: Cutlass code is heavily templated. Please make sure to fully trace through the entire dispatched path for the specific template specializations in these examples.  I want to understand the specialized call paths **end-to-end**.

**Remember**: The idea is to provide "literate code": a guided line-by-line walkthrough of the call stack / data flow with *annotated* inline code snippets and mappings to corresponding code sections (source file + line number spans).  

*Think pedagogic narrative + code + clickable links + visuals that guide the user, frame by frame, through the entire call graph, essentially a detailed guided tour of the system architecture with pointers for additional exploration.*

As a supplement, show how each of the above host / device templated objects / functions can be tested in isolation:
- provide minimal code snippets that can be run in isolation 
- goal of these minimal examples is to be able to poke into the internals and experiment with these APIs separate from the complexities of running examples end-to-end.

Assume a beginner knowledge of C++ -- that is, provide necessary background on C++ syntax / semantics / structure to the code base. 

Liberal use of markdown visuals (tables, diagrams, etc.) is encouraged.

Code links should be clickable from within the VScode editor (when viewed as a markdown doc): e.g.,
[barrier.cuh:196](src/include/non_abi/device/coll/barrier.cuh#L196)

Do all your work in a folder "claude/blackwell-gemm" at the repo root.