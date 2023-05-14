#!/usr/bin/env bash
. training_variables.sh

SAC_NUM_COLLECT_WORKERS=4

SAC_REVERB_CPUS=4
SAC_TRAIN_CPUS=4
SAC_COLLECT_WORKER_CPUS=1

SAC_PARAMS="tfagents_system/minitaur_params.py"
SAC_CONFIG="config/minitaur.json"