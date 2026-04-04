#!/usr/bin/env python3
"""Trace the live boundary between CuTeDSL wrappers and MLIR builders.

This script compiles a tiny ``@cute.jit`` function and prints probe events that
show how tracing moves through:

1. AST preprocessing
2. MLIR block-argument reconstruction into DSL wrapper objects
3. DSL arithmetic on proxy values
4. A direct call into ``cutlass._mlir.dialects.arith``

It does not execute a kernel. By default it sets ``CUTE_DSL_DRYRUN=1`` so the
run stops after IR generation.
"""

import functools
import os
from contextlib import ExitStack, contextmanager
from typing import Any, Callable

os.environ.setdefault("CUTE_DSL_DRYRUN", "1")
os.environ.setdefault("CUTE_DSL_NO_CACHE", "1")

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith as mlir_arith
from cutlass.base_dsl import ast_helpers as ast_helpers_mod
from cutlass.base_dsl import compiler as compiler_mod
from cutlass.base_dsl import dsl as dsl_mod
from cutlass.base_dsl import typing as typing_mod
from cutlass.base_dsl._mlir_helpers import arith as arith_helper

_EVENT_COUNTER = 0
_EVENT_DEPTH = 0


def _shorten(text: str, limit: int = 120) -> str:
    text = text.replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def summarize(value: Any) -> str:
    if isinstance(value, (str, int, float, bool, type(None))):
        return repr(value)
    if isinstance(value, ir.Value):
        return f"ir.Value({value}, type={value.type})"
    if isinstance(value, dict):
        pieces = []
        for key, item in list(value.items())[:3]:
            pieces.append(f"{key}={summarize(item)}")
        trailer = ", ..." if len(value) > 3 else ""
        return "{" + ", ".join(pieces) + trailer + "}"
    if isinstance(value, (list, tuple)):
        pieces = ", ".join(summarize(item) for item in list(value)[:3])
        trailer = ", ..." if len(value) > 3 else ""
        opener, closer = ("[", "]") if isinstance(value, list) else ("(", ")")
        return f"{opener}{pieces}{trailer}{closer}"
    if callable(value) and hasattr(value, "__name__"):
        return value.__name__
    if hasattr(value, "value"):
        inner = getattr(value, "value")
        return f"{type(value).__name__}(value={summarize(inner)})"
    if hasattr(value, "type"):
        return f"{type(value).__name__}(type={getattr(value, 'type', None)})"
    return _shorten(repr(value))


def emit_event(stage: str, detail: str = "") -> None:
    global _EVENT_COUNTER
    _EVENT_COUNTER += 1
    indent = "  " * _EVENT_DEPTH
    suffix = f": {detail}" if detail else ""
    print(f"[{_EVENT_COUNTER:02d}] {indent}{stage}{suffix}")


def note(label: str, value: Any | None = None) -> None:
    if value is None:
        print(f"[dsl] {label}")
    else:
        print(f"[dsl] {label}: {summarize(value)}")


@contextmanager
def patch_function(
    owner: Any,
    name: str,
    label: str,
    before: Callable[..., str] | None = None,
    after: Callable[..., str] | None = None,
):
    original = getattr(owner, name)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        global _EVENT_DEPTH

        detail = before(*args, **kwargs) if before else ""
        emit_event(f"enter {label}", detail)
        _EVENT_DEPTH += 1
        try:
            result = original(*args, **kwargs)
        except Exception as exc:
            _EVENT_DEPTH -= 1
            emit_event(f"raise {label}", f"{type(exc).__name__}: {_shorten(str(exc))}")
            raise
        _EVENT_DEPTH -= 1
        exit_detail = after(result, *args, **kwargs) if after else summarize(result)
        emit_event(f"exit {label}", exit_detail)
        return result

    setattr(owner, name, wrapped)
    try:
        yield
    finally:
        setattr(owner, name, original)


def install_probes(stack: ExitStack) -> None:
    stack.enter_context(
        patch_function(
            compiler_mod.CompileCallable,
            "_compile",
            "CompileCallable._compile",
            before=lambda self, func, *args, **kwargs: (
                f"func={func.__name__}, args={summarize(args)}, kwargs={summarize(kwargs)}"
            ),
        )
    )
    stack.enter_context(
        patch_function(
            dsl_mod.BaseDSL,
            "run_preprocessor",
            "BaseDSL.run_preprocessor",
            before=lambda self, original_function: (
                f"function={original_function.__name__}"
            ),
        )
    )
    stack.enter_context(
        patch_function(
            dsl_mod.BaseDSL,
            "generate_mlir_function_types",
            "BaseDSL.generate_mlir_function_types",
            before=lambda self, func_body, function_name, args, kwargs, args_spec, compile_only: (
                f"function_name={function_name}, args={summarize(args)}, annotations={summarize(args_spec.annotations)}"
            ),
        )
    )
    stack.enter_context(
        patch_function(
            dsl_mod.BaseDSL,
            "generate_execution_arguments",
            "BaseDSL.generate_execution_arguments",
            before=lambda self, args, kwargs, fop, args_spec: (
                f"template_args={summarize(args)}, block_arg_count={len(fop.regions[0].blocks[0].arguments)}"
            ),
        )
    )
    stack.enter_context(
        patch_function(
            dsl_mod,
            "new_from_mlir_values",
            "new_from_mlir_values",
            before=lambda obj, values: (
                f"template={summarize(obj)}, values={summarize(values)}"
            ),
        )
    )
    stack.enter_context(
        patch_function(
            typing_mod,
            "implicitDowncastNumericType",
            "implicitDowncastNumericType",
            before=lambda value: f"value={summarize(value)}",
        )
    )
    stack.enter_context(
        patch_function(
            ast_helpers_mod,
            "compare_executor",
            "compare_executor",
            before=lambda left, comparators, ops: (
                f"left={summarize(left)}, comparators={summarize(comparators)}, ops={summarize(ops)}"
            ),
        )
    )
    stack.enter_context(
        patch_function(
            ast_helpers_mod.executor,
            "_if_dynamic",
            "_if_execute_dynamic",
            before=lambda pred, then_block, else_block=None, mix_yield_args=None, full_write_args_count=0, mix_yield_arg_names=None, if_constexpr=None: (
                f"pred={summarize(pred)}, has_else={else_block is not None}"
            ),
        )
    )
    stack.enter_context(
        patch_function(
            arith_helper,
            "const",
            "arith_helper.const",
            before=lambda value, ty=None, **kwargs: (
                f"value={summarize(value)}, ty={summarize(ty)}"
            ),
        )
    )
    for opname in ("constant", "addi", "subi", "cmpi"):
        stack.enter_context(
            patch_function(
                mlir_arith,
                opname,
                f"mlir_arith.{opname}",
                before=lambda *args, _opname=opname, **kwargs: (
                    f"args={summarize(args)}, kwargs={summarize(kwargs)}"
                ),
            )
        )


@cute.jit
def boundary_demo(use_bias: cutlass.Constexpr, x: cutlass.Int32):
    note("function entry", x)

    if cutlass.const_expr(use_bias):
        note("constexpr branch", "Python chose bias = 3")
        bias = cutlass.Int32(3)
    else:
        note("constexpr branch", "Python chose bias = 1")
        bias = cutlass.Int32(1)

    y = x + bias
    note("DSL add result", y)

    raw_mlir_sum = mlir_arith.addi(y, cutlass.Int32(5))
    note("direct mlir arith.addi result", raw_mlir_sum)

    if y > 0:
        note("dynamic then body", y)
        _ = mlir_arith.subi(raw_mlir_sum, cutlass.Int32(2))
    else:
        note("dynamic else body", y)
        _ = mlir_arith.addi(raw_mlir_sum, cutlass.Int32(2))

    note("function exit", y)


def main() -> None:
    print("Tracing compile-time CuTeDSL -> MLIR interaction")
    print(f"cutlass package: {cutlass.__file__}")
    print(f"CUTE_DSL_DRYRUN={os.environ.get('CUTE_DSL_DRYRUN')}")
    print()

    with ExitStack() as stack:
        install_probes(stack)
        cute.compile(boundary_demo, True, 7)

    print()
    print("Done. This stopped after IR generation and never launched a kernel.")


if __name__ == "__main__":
    main()
