#!/usr/bin/env bash
cleanup() {
  # give other processes time to clean up
  sleep 10

  kill $OTHER_PIDS
  for OPID in $OTHER_PIDS; do
    wait $OPID
  done
  
  kill -INT $REPLAY_BUFFER_PID
  wait $REPLAY_BUFFER_PID
}

trap 'trap " " SIGTERM SIGINT; cleanup; wait' SIGINT SIGTERM EXIT

ALL_CPUS=$(cat /proc/self/status | grep Cpus_allowed_list | perl -pe 's/[a-z\s_:]//gi' | perl -pe 's/(\d+)-(\d+)/join(",", $1..$2)/ge' | perl -pe 's/,/ /g')
ALL_CPUS=( $ALL_CPUS )
CURR_IDX=0

OTHER_PIDS=""
# job steps
for (( idx=0;idx<$SAC_NUM_COLLECT_WORKERS;idx++ ))
do
  THIS_CPUS=$(echo "${ALL_CPUS[@]:$CURR_IDX:$SAC_COLLECT_WORKER_CPUS}" | perl -pe 's/ /,/g')
  SLURM_PROCID=$idx OMP_NUM_THREADS=$SAC_COLLECT_WORKER_CPUS taskset -c $THIS_CPUS \
  python tfagents_system/sac_collect.py --reverb_port $SAC_REVERB_PORT --output $SAC_EXP_DIR \
    --init_replay_buffer --seed $idx &
  CURR_IDX=$(( $CURR_IDX + $SAC_COLLECT_WORKER_CPUS ))
  OTHER_PIDS+="$! "
done

THIS_CPUS=$(echo "${ALL_CPUS[@]:$CURR_IDX:$SAC_TRAIN_CPUS}" | perl -pe 's/ /,/g')
OMP_NUM_THREADS=$SAC_TRAIN_CPUS taskset -c $THIS_CPUS \
python tfagents_system/sac_train.py --params $SAC_PARAMS --reverb_port $SAC_REVERB_PORT \
  --config $SAC_CONFIG --output $SAC_EXP_DIR &
CURR_IDX=$(( $CURR_IDX + $SAC_TRAIN_CPUS ))
OTHER_PIDS+="$! "

THIS_CPUS=$(echo "${ALL_CPUS[@]:$CURR_IDX:$SAC_REVERB_CPUS}" | perl -pe 's/ /,/g')
OMP_NUM_THREADS=$SAC_REVERB_CPUS OLD_TMPDIR=$TMPDIR TMPDIR=${SAC_EXP_DIR}/replay_buffer_tmp taskset -c $THIS_CPUS \
python tfagents_system/sac_replay_buffer.py --reverb_port $SAC_REVERB_PORT --output $SAC_EXP_DIR &
CURR_IDX=$(( $CURR_IDX + $SAC_REVERB_CPUS ))
REPLAY_BUFFER_PID=$!

wait -n
