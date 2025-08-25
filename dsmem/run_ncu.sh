#!/bin/bash
NCU=/home/jeromeku/cuda-toolkit/bin/ncu
SET=detailed
SECTIONS=PmSampling,PmSampling_WarpStates,WarpStateStats
SCRIPT=store_remote
PM_ARGS="--pm-sampling-interval 100000 --pm-sampling-buffer-size 67108864 --pm-sampling-max-passes 0"
WS_ARGS="--warp-sampling-interval auto --warp-sampling-buffer-size 67108864 warp-sampling-max-passes 5"
CLOCK="--clock-control base"
TENSOR_PIPE="pipeline-boost-state stable"
MP_ARGS="--target-processes all"
SECTION_REGEX="regex:.*(Sampling|Warp).*"

CMD="${NCU} \
  --set ${SET} \
  --section \"${SECTION_REGEX}\" \
  --import-source yes \
  -f \
  --verbose \
  -o ncu_${SCRIPT}.ncu-rep \
  ${SCRIPT}"

echo ">> ${CMD}"

mkdir -p temp
export TMPDIR=temp

eval ${CMD}