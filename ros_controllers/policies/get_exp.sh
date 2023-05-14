#!/usr/bin/env bash
if [ $# -ne 3 ]; then
  echo "Usage: $0 remote_host repo_root exp_name"
  echo "Got $# arguments instead"
  exit -1
fi

REMOTEDIR=`echo $2 | sed 's|'$HOME'|$HOME|'`/logs/$3

rsync --include='obs_keys.txt' --exclude={'__pycache__','replay_buffer_tmp','supervise','*.txt','events.out.tfevents.*'} \
  -mavzhe ssh $1:$REMOTEDIR .
