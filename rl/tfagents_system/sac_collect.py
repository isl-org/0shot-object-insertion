import init_paths
import argparse
import json
import logging
import numpy as np
import os
from tf_agents.train.utils import train_utils
from tfagents_system.train_params_base import TrainParamsBase
from tfagents_system.workers import Worker
from tfagents_system.utils.comp import config_tf
import time
from utilities import setup_logging, import_params_module, check_training_done


osp = os.path


def collect(exp_dir, ps: TrainParamsBase, reverb_port: int, init_replay_buffer: int, seed: int,
            logger: logging.Logger):
  # exit if training has finished
  train_step = train_utils.create_train_step()
  if check_training_done(exp_dir, train_step, ps):
    logger.info('Training done')
    return

  rng: np.random.Generator = np.random.default_rng(seed)
  seeds = rng.integers(1000, size=2)
  id = int(os.environ['SLURM_PROCID'])
  worker = Worker(id, exp_dir, ps, reverb_port, seeds)

  if init_replay_buffer and worker.train_step.numpy() == 0:
    logger.info('initial random collection start')
    worker.collect_random()
    logger.info('initial random collection done')
  worker.run()
  worker.close()
  logger.info('collection done')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--reverb_port', type=int, default=8008)
  parser.add_argument("-o", "--output", type=str, default="output", help="output folder")
  parser.add_argument('--init_replay_buffer', action='store_true',
                      help='Flag decides whether the SAC buffer is initialized with random steps')
  parser.add_argument('--seed', type=int, default=None)
  args = parser.parse_args()
  config_tf(1, int(os.environ['OMP_NUM_THREADS']))
  
  worker_id = int(os.environ["SLURM_PROCID"])
  log_filename = osp.join(args.output, f'collect_log_{worker_id}.txt')
  logger = setup_logging(filename=log_filename, level=logging.INFO, name=f'collect_{worker_id}')

  exp_config_filename = osp.join(args.output, 'config.json')
  exp_params_filename = osp.join(args.output, 'params.py')
  while not (osp.isfile(exp_config_filename) and osp.isfile(exp_params_filename)):
    logger.info(f'Waiting for {exp_config_filename} and {exp_params_filename}')
    time.sleep(1.0)

  params_module = import_params_module(exp_params_filename)
  with open(exp_config_filename, 'r') as f:
    config = json.load(f)
  ps: TrainParamsBase = params_module.TrainParams(config)

  # Perform collection.
  collect(args.output, ps, args.reverb_port, args.init_replay_buffer, args.seed, logger)
