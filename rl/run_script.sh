#!/usr/bin/env bash
if [ "$#" -ne 1 ]; then
    echo "usage: ./run_script <training_variables (.sh) file>"
    exit -1
fi

set -a
. $1
set +a

# check if enough CPUs are available
NEEDED_CPUS=$(( $SAC_REVERB_CPUS + $SAC_TRAIN_CPUS + $SAC_NUM_COLLECT_WORKERS * $SAC_COLLECT_WORKER_CPUS ))
if [ $(nproc) -lt $NEEDED_CPUS ]
then
  echo "Need $NEEDED_CPUS CPUs, only $(nproc) available"
  exit -1
fi

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

if [ -z ${SLURM_ARRAY_TASK_ID} ]
then
  export SAC_EXP_DIR=logs/$SAC_EXP_NAME
else
  export SAC_EXP_DIR=logs/${SAC_EXP_NAME}_task_${SLURM_ARRAY_TASK_ID}
fi
mkdir -p $SAC_EXP_DIR/replay_buffer_tmp
cp run $SAC_EXP_DIR
supervise $SAC_EXP_DIR
