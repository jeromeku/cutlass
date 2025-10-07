from cutlass_logging import set_logging, patch_cutlass_loggers
from types import ModuleType
# set_logging(log_to_console=True)
patch_cutlass_loggers()

import cutlass._mlir._mlir_libs as libs

for name in dir(libs):
    obj = getattr(libs, name)
    if isinstance(obj, ModuleType) and name.startswith("_") and "__" not in name:
        print("---------------------------------------")
        print(obj)
        for _attr in dir(obj):
            print(f"{_attr}: {getattr(obj, _attr)}")
        print("---------------------------------------")