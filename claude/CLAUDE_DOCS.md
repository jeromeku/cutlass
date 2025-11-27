Write an Architecture.md for this repo.

See this link for how to create an Architecture.md file that describe system architecture (https://matklad.github.io/2021/02/06/ARCHITECTURE.md.html).  


Specifically, I am interested in:
- how do the various components within `max` -- `_core`, `_core_mojo`, `kernels`, `graph`, `serve`, `engine`, `driver`, `compiler` -- fit together?  Help me build a mental map of the entire mojo / modular stack and where each component sits in this stack and what role each plays.
- `mojo`'s compiler pipeline
    - How is mojo code lowered to hardware-specific binaries?
    - Trace the entire process from user mojo code (*.mojo) to AST parsing -> MLIR dialects -> MLIR passes -> codegen -> binary
    - How is this binary then bound to either mojo or other languages?
    - Which parts of this pipeline are open sourced?
- `mojo` < - > `MLIR` interop
    - How does `mojo` interoperate with `MLIR`?
    - Which parts of the `MLIR` infra does `mojo` use?
    - How does it modify / extend `MLIR`?
- `mojo` < - > `python` interop
    - How can `python` code be called from `mojo` and vice versa?
    - How are these bindings implemented?

The idea is to provide "literate code": a guided line-by-line walkthrough of the call stack / data flow with *annotated* inline code snippets and mappings to corresponding code sections (source file + line number spans).  

*Think pedagogic narrative + code + clickable links + visuals that guide the user, frame by frame, through the entire call graph, essentially a detailed guided tour of the system architecture with pointers for additional exploration.*

Liberal use of markdown visuals (tables, diagrams, etc.) is encouraged.

Code links should be clickable from within the VScode editor (when viewed as a markdown doc): e.g.,
[barrier.cuh:196](src/include/non_abi/device/coll/barrier.cuh#L196)
 
Do all your work in a folder "claude/docs" at the repo root.