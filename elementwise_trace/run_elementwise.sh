export CUTE_DSL_DISABLE_FILE_CACHING=1
export CUTE_DSL_LOG_LEVEL=10 # 10: DEBUG
export CUTE_DSL_PRINT_AFTER_PREPROCESSOR=1
# export CUTE_DSL_DRYRUN=1
export CUTE_DSL_LOG_TO_FILE=0
export CUTE_DSL_LOG_TO_CONSOLE=1
LOGDIR="logs"
mkdir -p ${LOGDIR}
dt=`date '+%Y%m%d_%H%M'`
RECORD_LOG="0"
CMD="python elementwise_apply.py --skip_ref_check"
echo ${CMD}

if [[ ${RECORD_LOG} == "1" ]]; then
    CMD="${CMD} 2>&1 | tee ${LOGDIR}/${dt}_elementwise.log"
fi

eval ${CMD}

