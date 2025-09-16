# save as extract_ptx.py
from cutlass._mlir import ir
import re, codecs

MLIR_FILE = "builtin_module_no-symbol-name/35_external-kernel-for-gpu-launch.mlir"
with open(MLIR_FILE, "r") as f:
    txt = f.read()

with ir.Context() as ctx:
    m = ir.Module.parse(txt)
    ptx_blobs = []
    def visit(op):
        if op.name == "gpu.binary":
            # generic access to attrs; "objects" is an ArrayAttr of #gpu.object
            arr = op.attributes["objects"]
            for obj in arr:
                s = str(obj)  # prints like: #gpu.object<#nvvm.target<...>, assembly = "..." >
                # crude but robust: pull the assembly="..." payload
                m = re.search(r'assembly\s*=\s*"([^"]*)"', s)
                if m:
                    # Unescape \0A etc. MLIR prints C-style escapes.
                    raw = m.group(1).encode("utf-8").decode("unicode_escape")
                    ptx_blobs.append(raw)
        for o in op.regions:
            for b in o.blocks:
                for inner in b.operations:
                    visit(inner)
    visit(m.operation)

def mlir_assembly_to_ptx(s: str) -> str:
    # MLIR prints C-style escapes, but you'll often see octet sequences like \0A (LF) and \09 (TAB).
    # Make them real newlines/tabs and drop trailing NULs.
    s = s.replace("\x00A", "\n").replace("\x009", "\t").replace("\x00", "")
    # If you have other hex-ish escapes, map them here as needed.
    return s

# Write one PTX file per object
for i, ptx in enumerate(ptx_blobs):
    with open(f"kernel_{i}.ptx", "w") as out:
        ptx = mlir_assembly_to_ptx(ptx)
        breakpoint()
        out.write(ptx)
print(f"Wrote {len(ptx_blobs)} PTX file(s).")
