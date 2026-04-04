# Plan: visibility across CuTeDSL Python, MLIR, and kernel execution

1. Identify the built-in CuTeDSL visibility hooks already present in the repo: IR printing, IR/PTX/CUBIN dumps, lineinfo, and JIT timing.
2. Separate the workflow into two phases:
   - compile time: Python execution, AST preprocessing, tracing, MLIR emission, pass pipeline
   - runtime: host launch path, CUDA timeline, kernel execution on device
3. Compare tools by layer:
   - `py-spy` for Python-side compile-time behavior
   - CuTeDSL dump/log knobs for MLIR/PTX/CUBIN visibility
   - Nsight Systems / Nsight Compute / Compute Sanitizer for runtime GPU visibility
4. Write a repo-root note with exact source links and a practical “what to use for what question” summary.
