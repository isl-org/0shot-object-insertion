#!/bin/sh

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
export MUJOCO_PY_MJKEY_PATH=${HOME}/mjkey.txt
export MUJOCO_PY_MUJOCO_PATH=${HOME}/.mujoco/mujoco200_linux
export OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MUJOCO_PY_MUJOCO_PATH}/bin:${CONDA_PREFIX}/lib
