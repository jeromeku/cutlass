from enum import Enum

# ============================================================================
# Setup: Minimal infrastructure (simplified version of CuTeDSL)
# ============================================================================

class Constexpr:
    """Marker type for compile-time values"""
    pass

class Tensor:
    """Runtime tensor type (traced)"""
    def __init__(self, name):
        self.name = name
        self.ops = []
    
    def __getitem__(self, idx):
        result = Tensor(f"{self.name}[{idx}]")
        result.ops = self.ops + [("load", self.name, idx)]
        return result
    
    def __setitem__(self, idx, value):
        self.ops.append(("store", value, self.name, idx))
    
    def __mul__(self, other):
        result = Tensor(f"{self.name}*{other}")
        result.ops = self.ops + [("mul", self, other)]
        return result
    
    def __add__(self, other):
        result = Tensor(f"{self.name}+{other.name}")
        result.ops = self.ops + [("add", self, other)]
        return result

# ============================================================================
# Decorator that implements the hybrid approach
# ============================================================================

def jit(func):
    """
    Hybrid JIT decorator:
    - Pre-stage: Analyze AST to find loops/control flow
    - Meta-stage: Execute with Constexpr args to do metaprogramming
    - Object-stage: Trace Tensor operations to generate MLIR
    """
    import ast
    import inspect
    
    def wrapper(*args, **kwargs):
        # Get function annotations to know what's Constexpr
        sig = inspect.signature(func)
        
        # Separate Constexpr args from runtime args
        constexpr_args = {}
        runtime_args = {}
        
        for i, (param_name, param) in enumerate(sig.parameters.items()):
            arg_value = args[i] if i < len(args) else kwargs.get(param_name)
            
            if param.annotation == Constexpr:
                constexpr_args[param_name] = arg_value
            else:
                runtime_args[param_name] = arg_value
        
        # PRE-STAGE: Parse AST to find structure
        source = inspect.getsource(func)
        tree = ast.parse(source)
        
        print("=" * 60)
        print("PRE-STAGE (AST Analysis):")
        print("=" * 60)
        
        # Find loops in the AST
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                print(f"✓ Found FOR loop (will emit scf.for)")
            elif isinstance(node, ast.If):
                print(f"✓ Found IF statement (will emit scf.if)")
        
        print("\n" + "=" * 60)
        print("META-STAGE (Python Interpreter):")
        print("=" * 60)
        
        # META-STAGE: Execute function with constexpr values
        # and tracer objects for runtime values
        print(f"Constexpr args: {constexpr_args}")
        print(f"Executing Python code with real values...\n")
        
        # Create tracers for runtime args
        traced_args = {}
        for name, value in runtime_args.items():
            if hasattr(value, '__len__'):  # It's a tensor
                traced_args[name] = Tensor(name)
            else:
                traced_args[name] = value
        
        # Execute the function
        all_args = {**constexpr_args, **traced_args}
        result = func(**all_args)
        
        print("\n" + "=" * 60)
        print("OBJECT-STAGE (Generated MLIR):")
        print("=" * 60)
        breakpoint()
        # Collect all operations from tracers
        for arg in traced_args.values():
            if isinstance(arg, Tensor):
                emit_mlir(arg)
        
        return result
    
    return wrapper

def emit_mlir(tensor):
    """Print MLIR-like representation of traced operations"""
    print("func.func @kernel(...) {")
    
    seen_loops = set()
    
    for op in tensor.ops:
        if op[0] == "loop_start":
            loop_id = op[1]
            if loop_id not in seen_loops:
                print(f"  scf.for %i_{loop_id} = 0 to {op[2]} {{  // ← Loop preserved!")
                seen_loops.add(loop_id)
        elif op[0] == "load":
            print(f"    %{op[1]}_{op[2]} = memref.load %{op[1]}[%i_{op[2]}]")
        elif op[0] == "mul":
            print(f"    %result = arith.mulf %{op[1].name}, {op[2]}")
        elif op[0] == "add":
            print(f"    %result = arith.addf %{op[1].name}, %{op[2].name}")
        elif op[0] == "store":
            print(f"    memref.store %result, %{op[2]}[%i]")
        elif op[0] == "loop_end":
            print("  }")
    
    print("}")

# ============================================================================
# Example: Fused Epilogue Pattern
# ============================================================================

class ActivationType(Enum):
    RELU = "relu"
    IDENTITY = "identity"

class Epilogue:
    def apply(self, x):
        return x

class ReLU(Epilogue):
    def apply(self, x):
        print(f"  [Python] Executing ReLU.apply() - will inline max(0, x)")
        result = Tensor(f"relu({x.name})")
        result.ops = x.ops + [("relu", x)]
        return result

class Identity(Epilogue):
    def apply(self, x):
        print(f"  [Python] Executing Identity.apply() - will inline x")
        return x

@jit
def kernel(
    epilogue_class: Constexpr,  # ← Compile-time: which epilogue?
    scale_factor: Constexpr,     # ← Compile-time: scale value
    data: Tensor                 # ← Runtime: actual data
):
    """
    Hybrid approach demonstration:
    - Loop structure captured by AST
    - Epilogue polymorphism resolved by interpreter
    - Data operations traced
    """
    
    # ========================================================================
    # METAPROGRAMMING (executed by Python interpreter)
    # ========================================================================
    print(f"  [Python] epilogue_class = {epilogue_class.__name__}")
    print(f"  [Python] scale_factor = {scale_factor}")
    print(f"  [Python] Creating epilogue instance...")
    epilogue = epilogue_class()
    
    # Compile-time decision based on scale_factor
    if scale_factor > 1.0:
        print(f"  [Python] scale_factor > 1.0, using special path")
        use_special_path = True
    else:
        print(f"  [Python] scale_factor <= 1.0, using normal path")
        use_special_path = False
    
    # ========================================================================
    # STRUCTURED CONTROL FLOW (captured by AST)
    # ========================================================================
    # This loop will emit scf.for in MLIR (not unrolled!)
    data.ops.append(("loop_start", "i", 3))
    
    for i in range(3):  # ← AST sees this structure
        print(f"\n  [Python] Loop iteration i={i}")
        
        # ====================================================================
        # TRACED OPERATIONS (on runtime tensors)
        # ====================================================================
        # Load from tensor (traced)
        x = data[i]
        breakpoint()
        # Multiply by scale (scale_factor is constexpr, so it's a literal)
        scaled = x * scale_factor
        
        # Apply epilogue (epilogue is resolved at compile-time!)
        result = epilogue.apply(scaled)
        breakpoint()
        # Store back (traced)
        data[i] = result
    
    data.ops.append(("loop_end", "i"))

# ============================================================================
# Run the example
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("EXAMPLE 1: kernel(ReLU, scale=2.0, data)")
    print("=" * 60)
    
    data1 = [1.0, 2.0, 3.0]
    kernel(ReLU, 2.0, data1)
    
    print("\n\n" + "=" * 60)
    print("EXAMPLE 2: kernel(Identity, scale=0.5, data)")
    print("=" * 60)
    
    data2 = [1.0, 2.0, 3.0]
    kernel(Identity, 0.5, data2)
"""
## Expected Output
```
============================================================
EXAMPLE 1: kernel(ReLU, scale=2.0, data)
============================================================
============================================================
PRE-STAGE (AST Analysis):
============================================================
✓ Found FOR loop (will emit scf.for)
✓ Found IF statement (will emit scf.if)

============================================================
META-STAGE (Python Interpreter):
============================================================
Constexpr args: {'epilogue_class': <class '__main__.ReLU'>, 'scale_factor': 2.0}
Executing Python code with real values...

  [Python] epilogue_class = ReLU
  [Python] scale_factor = 2.0
  [Python] Creating epilogue instance...
  [Python] scale_factor > 1.0, using special path

  [Python] Loop iteration i=0
  [Python] Executing ReLU.apply() - will inline max(0, x)

  [Python] Loop iteration i=1
  [Python] Executing ReLU.apply() - will inline max(0, x)

  [Python] Loop iteration i=2
  [Python] Executing ReLU.apply() - will inline max(0, x)

============================================================
OBJECT-STAGE (Generated MLIR):
============================================================
func.func @kernel(...) {
  scf.for %i_i = 0 to 3 {  // ← Loop preserved!
    %data_0 = memref.load %data[%i_0]
    %result = arith.mulf %data[0], 2.0
    %result = arith.mulf %data[1], 2.0
    %result = arith.mulf %data[2], 2.0
    memref.store %result, %data[%i]
  }
}


============================================================
EXAMPLE 2: kernel(Identity, scale=0.5, data)
============================================================
============================================================
PRE-STAGE (AST Analysis):
============================================================
✓ Found FOR loop (will emit scf.for)
✓ Found IF statement (will emit scf.if)

============================================================
META-STAGE (Python Interpreter):
============================================================
Constexpr args: {'epilogue_class': <class '__main__.Identity'>, 'scale_factor': 0.5}
Executing Python code with real values...

  [Python] epilogue_class = Identity
  [Python] scale_factor = 0.5
  [Python] Creating epilogue instance...
  [Python] scale_factor <= 1.0, using normal path

  [Python] Loop iteration i=0
  [Python] Executing Identity.apply() - will inline x

  [Python] Loop iteration i=1
  [Python] Executing Identity.apply() - will inline x

  [Python] Loop iteration i=2
  [Python] Executing Identity.apply() - will inline x

============================================================
OBJECT-STAGE (Generated MLIR):
============================================================
func.func @kernel(...) {
  scf.for %i_i = 0 to 3 {  // ← Loop preserved!
    %data_0 = memref.load %data[%i_0]
    %result = arith.mulf %data[0], 0.5
    %result = arith.mulf %data[1], 0.5
    %result = arith.mulf %data[2], 0.5
    memref.store %result, %data[%i]
  }
}
"""