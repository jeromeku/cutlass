
timeit() {
  local START END
  START=$(date +%s.%N)
  "$@"
  END=$(date +%s.%N)
  ELAPSED=$(echo "$END - $START" | bc)
  echo "Elapsed time: ${ELAPSED} seconds"
}