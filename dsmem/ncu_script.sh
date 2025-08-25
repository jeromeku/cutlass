#!/bin/bash

# Configuration
NCU="/home/jeromeku/cuda-toolkit/bin/ncu"
SCRIPT="store_remote"
SET="detailed"
SECTION_REGEX="regex:.*(Sampling|Warp).*"
TS=$(date -u +%Y%m%dT%H%M)

# Core arguments array
declare -a ARGS=(
    "--set" "${SET}"
    "--section" "${SECTION_REGEX}"
    "--import-source" "yes"
    "-f"
    "--verbose"
    "-o" "ncu_${SCRIPT}_${TS}.ncu-rep"
)

# Feature flags
ENABLE_CALLSTACK=1
ENABLE_PM_SAMPLING=0
ENABLE_WARP_SAMPLING=0
ENABLE_CLOCK_CONTROL=0
ENABLE_TENSOR_PIPE=0
ENABLE_MP_ARGS=0

# Conditional arguments based on feature flags
if [[ ${ENABLE_CALLSTACK} -eq 1 ]]; then
    ARGS+=("--call-stack" "yes")
    ARGS+=("--call-stack-type" "native")
    ARGS+=("--call-stack-type" "python")
fi

PM_SAMPLING=0
PM_BUF_SIZE=0
PM_PASSES=0
if [[ ${ENABLE_PM_SAMPLING} -eq 1 ]]; then
    ARGS+=("--pm-sampling-interval" "${PM_SAMPLING}")
    ARGS+=("--pm-sampling-buffer-size" "${PM_BUF_SIZE}")
    ARGS+=("--pm-sampling-max-passes" "${PM_PASSES}")
fi

WS_SAMPLING=0
WS_BUF_SIZE=0
WS_PASSES=5
if [[ ${ENABLE_WARP_SAMPLING} -eq 1 ]]; then
    ARGS+=("--warp-sampling-interval" "${WS_SAMPLING}")
    ARGS+=("--warp-sampling-buffer-size" "${WS_BUF_SIZE}")
    ARGS+=("--warp-sampling-max-passes" "${WS_PASSES}")
fi

if [[ ${ENABLE_CLOCK_CONTROL} -eq 1 ]]; then
    ARGS+=("--clock-control" "base")
fi

if [[ ${ENABLE_TENSOR_PIPE} -eq 1 ]]; then
    ARGS+=("pipeline-boost-state" "stable")
fi

if [[ ${ENABLE_MP_ARGS} -eq 1 ]]; then
    ARGS+=("--target-processes" "all")
fi

# Add the script to profile as the final argument
ARGS+=("${SCRIPT}")

# Build and execute command
echo ">> ${NCU} ${ARGS[*]}"

mkdir -p temp
export TMPDIR=temp

# Execute with array expansion
"${NCU}" "${ARGS[@]}"