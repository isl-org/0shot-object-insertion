#!/usr/bin/env bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 ROBOSUITE_CLONE_DIR"
  exit -1
fi

. setup_env_vars.sh

python -m pip install mujoco-py==2.0.2.9 && python -m pip uninstall mujoco-py && python -m pip install mujoco-py==2.0.2.9 --no-cache-dir \
  --no-binary :all: --no-build-isolation
python -m pip install -e $1
