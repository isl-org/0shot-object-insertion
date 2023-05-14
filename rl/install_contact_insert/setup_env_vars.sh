#!/usr/bin/env bash
eval "$(conda shell.bash hook)"
conda activate sac_utils

mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cp activate_env_vars.sh $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
cp deactivate_env_vars.sh $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh

conda deactivate
conda activate sac_utils
