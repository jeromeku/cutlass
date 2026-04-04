#!/bin/bash
SUBPROCESS=${SUBPROCESS:-0}
OUTPUT=${OUTPUT:-speedscope}
ARGS=("
    --native
    --full-filenames
    --subprocesses
    -f ${OUTPUT}
    -o dsl-trace.json
")

CMD="py-spy record ${ARGS[@]} $@"
echo $CMD
eval ${CMD}