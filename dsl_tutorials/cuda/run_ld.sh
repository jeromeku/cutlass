#!/bin/bash

set -euo pipefail

OPTIONS="bindings"
PROGRAM="test_cuda.py"
LD_LOG=ld_${OPTIONS}.log

CMD="LD_DEBUG=${OPTIONS} LD_BIND_NOW=1 LD_DEBUG_OUTPUT=${LD_LOG} python $PROGRAM"
echo ${CMD}
eval ${CMD}