import ast
import inspect
import textwrap

# ------------------------------------------------------------
# 1. AST-BASED LOWERING
# ------------------------------------------------------------

class ASTLowerer(ast.NodeVisitor):
    def __init__(self):
        self.indent = 0

    def p(self, msg):
        print("  " * self.indent + msg)

    def visit_FunctionDef(self, node):
        args = [a.arg for a in node.args.args]
        self.p(f"func {node.name}({', '.join(args)}):")
        self.indent += 1
        for stmt in node.body:
            self.visit(stmt)
        self.indent -= 1

    def visit_For(self, node):
        target = ast.unparse(node.target)
        iterable = ast.unparse(node.iter)
        self.p(f"for {target} in {iterable}:")
        self.indent += 1
        for stmt in node.body:
            self.visit(stmt)
        self.indent -= 1

    def visit_Assign(self, node):
        targets = ", ".join(ast.unparse(t) for t in node.targets)
        value = ast.unparse(node.value)
        self.p(f"{targets} = {value}")

    def visit_Return(self, node):
        self.p(f"return {ast.unparse(node.value)}")


def kernel_ast(n):
    s = 0
    for i in range(n):
        s = s + i
    return s


def demo_ast():
    print("=== AST-based lowering ===")
    src = textwrap.dedent(inspect.getsource(kernel_ast))
    tree = ast.parse(src)
    fn_node = tree.body[0]  # kernel_ast
    ASTLowerer().visit(fn_node)


# ------------------------------------------------------------
# 2. TRACING-BASED LOWERING (LOSES LOOP)
# ------------------------------------------------------------

class TraceValue:
    def __init__(self, name):
        self.name = name

    def __add__(self, other):
        global trace_ir
        result_name = f"t{len(trace_ir)}"
        trace_ir.append(f"{result_name} = add {self.name}, {other.name}")
        return TraceValue(result_name)


def kernel_trace(n):
    s = TraceValue("s0")
    for i in range(n):
        i_val = TraceValue(f"i{i}")  # pretend i is a symbolic value
        s = s + i_val
    return s


def demo_trace():
    print("\n=== Tracing-based lowering (n=3) ===")
    global trace_ir
    trace_ir = []
    kernel_trace(3)
    for op in trace_ir:
        print(op)


# ------------------------------------------------------------
# 3. HYBRID LOWERING: dynamic loop + constexpr loop
# ------------------------------------------------------------

def range_dynamic(n):
    # stand-in for a loop whose bound we want in IR as a real loop
    return range(n)

def range_constexpr(n):
    # in CuTeDSL this would be a "constexpr" range evaluated at capture time
    return range(n)


class HybridLowerer(ast.NodeVisitor):
    def __init__(self):
        self.indent = 0
        self.ops = []
    def p(self, msg):
        print("  " * self.indent + msg)
        self.ops.append(msg)
    
    def visit_FunctionDef(self, node):
        args = [a.arg for a in node.args.args]
        self.p(f"func {node.name}({', '.join(args)}):")
        self.indent += 1
        for stmt in node.body:
            self.visit(stmt)
        self.indent -= 1

    def visit_For(self, node):
        # Decide: dynamic loop or constexpr loop?
        if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
            fname = node.iter.func.id
        else:
            fname = None
        
        if fname == "range_dynamic":
            # keep as control-flow in IR
            target = ast.unparse(node.target)
            iterable = ast.unparse(node.iter.args[0])
            self.p(f"for {target} in range({iterable}):  # dynamic loop -> scf.for")
            self.indent += 1
            for stmt in node.body:
                self.visit(stmt)
            self.indent -= 1

        elif fname == "range_constexpr":
            # emulate "meta-stage": unroll at compile time
            assert len(node.iter.args) == 1 and isinstance(node.iter.args[0], ast.Constant)
            bound = node.iter.args[0].value
            self.p(f"# constexpr loop over {bound} iterations, unrolled:")
            for i in range(bound):
                self.indent += 1
                self.p(f"# iteration {i}")
                # We could substitute i into the body; for simplicity just visit it.
                for stmt in node.body:
                    self.visit(stmt)
                self.indent -= 1
        else:
            self.p("# UNKNOWN for-loop kind")
            
            induction_var = ast.unparse(node.target)
            iterable = ast.unparse(node.iter)
            self.p(f"for {induction_var} in {iterable}")
            self.generic_visit(node)

    def visit_Assign(self, node):
        targets = ", ".join(ast.unparse(t) for t in node.targets)
        value = ast.unparse(node.value)
        self.p(f"{targets} = {value}")

    def visit_Return(self, node):
        self.p(f"return {ast.unparse(node.value)}")


def kernel_hybrid(n):
    s = 0
    # dynamic loop: we want a real loop in MLIR (e.g., scf.for)
    for i in range_dynamic(n):
        s = s + i

    # constexpr loop: we want Python to unroll it at capture time
    for j in range_constexpr(3):
        s = s + j

    for k in range(5):
        s = s + 1

    return s


def demo_hybrid():
    print("\n=== Hybrid lowering (dynamic + constexpr) ===")
    src = textwrap.dedent(inspect.getsource(kernel_hybrid))
    tree = ast.parse(src)
    fn_node = tree.body[0]
    hybrid = HybridLowerer()
    hybrid.visit(fn_node)
    print('\n'.join(hybrid.ops))

if __name__ == "__main__":
    demo_ast()
    demo_trace()
    demo_hybrid()
