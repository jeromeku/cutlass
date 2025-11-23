import sys

def example_function(arg1, arg2):
    local_var = 42
    # Get the current frame
    frame = sys._getframe()
    
    print("Frame information:")
    print(f"  Function name: {frame.f_code.co_name}")
    print(f"  Locals: {frame.f_locals}")
    print(f"  Globals available: {len(frame.f_globals)} items")
    print(f"  Bytecode instruction pointer: {frame.f_lasti}")
    print(f"  Line number: {frame.f_lineno}")
    print(f"  Previous frame: {frame.f_back.f_code.co_name if frame.f_back else None}")
    print(f"  Line number: {frame.f_lineno}")
    return "done"

frame = sys._getframe()
print("Frame information:")
print(f"  Function name: {frame.f_code.co_name}")
print(f"  Locals: {frame.f_locals}")
print(f"  Globals available: {len(frame.f_globals)} items")
print(f"  Bytecode instruction pointer: {frame.f_lasti}")
print(f"  Line number: {frame.f_lineno}")
print(f"  Previous frame: {frame.f_back.f_code.co_name if frame.f_back else None}")

example_function(1, 2)
