#!/bin/bash

set -euo pipefail
export CUTE_DSL_KEEP_CUBIN=1
export CUTE_DSL_KEEP_PTX=1
export CUTE_DSL_LINEINFO=1
export CUTE_DSL_DUMP_DIR="cute_dump"
export CUTE_DSL_FILTER_STACKTRACE=0
export CUTE_DSL_KEEP_IR=1
export CUTE_DSL_DISABLE_FILE_CACHING=1
export CUTE_DSL_LOG_LEVEL=10 # 10: DEBUG
export CUTE_DSL_PRINT_AFTER_PREPROCESSOR=1
# export CUTE_DSL_DRYRUN=1
export CUTE_DSL_LOG_TO_FILE=0
export CUTE_DSL_LOG_TO_CONSOLE=1
PROGRAM=$1

LOGDIR="logs"
mkdir -p ${LOGDIR}
dt=`date '+%Y%m%d_%H%M'`

RECORD_LOG="1"
CMD="python ${PROGRAM}"

echo ${CMD}

if [[ ${RECORD_LOG} == "1" ]]; then
    CMD="${CMD} 2>&1 | tee ${LOGDIR}/${dt}_elementwise.log"
fi

eval ${CMD}

