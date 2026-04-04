# Plan: explain `cpasync_reduce_bulk_add_f32` inline asm lowering

1. Inspect the inline-asm call in `thirdparty/quack/quack/copy_utils.py` and identify the exact operand and constraint mapping.
2. Trace how CuteDSL numeric and pointer wrapper objects expose raw MLIR/LLVM values via `ir_value()`.
3. Check the generated LLVM dialect bindings and nearby examples to explain `llvm.inline_asm(...)` and `llvm.AsmDialect.AD_ATT`.
4. Write a short trace note in `codex/` with clickable source links and summarize the reasoning in the terminal reply.
