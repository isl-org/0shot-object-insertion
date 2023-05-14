#!/usr/bin/env bash
eval "$(conda shell.bash hook)"
conda activate sac_utils

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

KMP_AFFINITY=granularity=fine,compact,1,0
KMP_BLOCKTIME=1
KMP_SETTINGS=TRUE

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

SAC_REVERB_PORT=`python3 -m portpicker $$`

# do not modify above, do modify below

# number of parallel workers that will collect exeprience and evaluate the current policy
SAC_NUM_COLLECT_WORKERS=20
# number of CPUs allocated to the reverb replay buffer program tfagents_system/sac_replay_buffer.py
SAC_REVERB_CPUS=8
# number of CPUs allocated to the policy update training program tfagents_system/sac_train.py
SAC_TRAIN_CPUS=8
# number of CPUs allocated to each worker tfagents_system/sac_collect.py
SAC_COLLECT_WORKER_CPUS=4
# name of the root experiment directory within the "logs" directory - it will store checkpoints and logs
SAC_EXP_NAME="develop"
# Python module that instantiates the TrainParamsBase abstract class from tfagents_system/train_params_base.py
SAC_PARAMS="tfagents_system/default_params.py"
# JSON containing configuration parameters for the RL environments used by workers. Available as self.env_config
# in TrainParamsBase tfagents_system/train_params_base.py
SAC_CONFIG="config/simple.json"