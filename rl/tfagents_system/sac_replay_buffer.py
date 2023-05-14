import init_paths
import argparse
import json
import logging
import reverb
import tensorflow as tf
from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.train import learner
from tf_agents.train.utils import train_utils
from tfagents_system.train_params_base import TrainParamsBase
import time
from utilities import setup_logging, import_params_module, check_training_done
import os


osp = os.path


def main(exp_dir, ps: TrainParamsBase, reverb_port, logger: logging.Logger):
  # exit if training has finished
  train_step = train_utils.create_train_step()
  if check_training_done(exp_dir, train_step, ps):
    logger.info('Training done')
    return

  ckpt_path_filename = osp.join(exp_dir, 'replay_buffer_ckpt_path.txt')
  try:
    with open(ckpt_path_filename, 'r') as f:
      lines = [l.strip() for l in f]
    fallback_ckpt_path = lines[0]
    ckpt_path, _ = osp.split(fallback_ckpt_path)
    checkpointer = reverb.checkpointers.DefaultCheckpointer(ckpt_path, fallback_checkpoint_path=fallback_ckpt_path)
  except FileNotFoundError:
    checkpointer = None 

  # Wait for the collect policy to become available, then load it.
  collect_policy_dir = osp.join(exp_dir, learner.POLICY_SAVED_MODEL_DIR, learner.COLLECT_POLICY_SAVED_MODEL_DIR)
  collect_policy = train_utils.wait_for_policy(collect_policy_dir, load_specs_from_pbtxt=True)

  # Create the signature for the variable container holding the policy weights.
  variables = {
      reverb_variable_container.POLICY_KEY: collect_policy.variables(),
      reverb_variable_container.TRAIN_STEP_KEY: train_step
  }
  variable_container_signature = tf.nest.map_structure(
      lambda variable: tf.TensorSpec(variable.shape, dtype=variable.dtype), variables)
  logger.info('Signature of variables: \n%s', variable_container_signature)

  # Create the signature for the replay buffer holding observed experience.
  replay_buffer_signature = tensor_spec.from_spec(collect_policy.collect_data_spec)
  replay_buffer_signature = tensor_spec.add_outer_dim(replay_buffer_signature)
  logger.info('Signature of experience: \n%s', replay_buffer_signature)

  # Create and start the replay buffer and variable container server.
  n = ps.compute_config['n_parallel_workers']
  server = reverb.Server(
      tables=[
          reverb.Table(  # Replay buffer storing experience.
              name=reverb_replay_buffer.DEFAULT_TABLE,
              sampler=reverb.selectors.Uniform(),
              remover=reverb.selectors.Fifo(),
              rate_limiter=reverb.rate_limiters.MinSize(1),
              max_size=ps.env_config['horizon'] * ps.n_episodes_in_buffer,
              max_times_sampled=0,
              signature=replay_buffer_signature,
          ),
          reverb.Table(  # Variable container storing policy parameters.
              name=reverb_variable_container.DEFAULT_TABLE,
              sampler=reverb.selectors.Uniform(),
              remover=reverb.selectors.Fifo(),
              rate_limiter=reverb.rate_limiters.SampleToInsertRatio(samples_per_insert=n, min_size_to_sample=1,
                                                                    error_buffer=n),
              max_size=1,
              max_times_sampled=n,
              signature=variable_container_signature,
          ),
      ],
      port=reverb_port,
      checkpointer=checkpointer,
  )
  server.wait()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--reverb_port', type=int, default=8008)
  parser.add_argument("-o", "--output", type=str, default="output", help="output folder")
  args = parser.parse_args()
  
  log_filename = osp.join(args.output, 'replay_buffer_log.txt')
  logger = setup_logging(filename=log_filename, level=logging.INFO, name="replay_buffer")

  exp_config_filename = osp.join(args.output, 'config.json')
  exp_params_filename = osp.join(args.output, 'params.py')
  while not (osp.isfile(exp_config_filename) and osp.isfile(exp_params_filename)):
    logger.info(f'Waiting for {exp_config_filename} and {exp_params_filename}')
    time.sleep(1.0)

  params_module = import_params_module(exp_params_filename)
  with open(exp_config_filename, 'r') as f:
    config = json.load(f)
  ps: TrainParamsBase = params_module.TrainParams(config)

  main(args.output, ps, args.reverb_port, logger)
