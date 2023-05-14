import init_paths
import argparse
import json
import logging
import os
import reverb
from shutil import copyfile
import tensorflow as tf
from tf_agents.agents.tf_agent import LossInfo
from tf_agents.experimental.distributed import reverb_variable_container, ReverbVariableContainer
from tf_agents.replay_buffers import reverb_replay_buffer, ReverbReplayBuffer
from tf_agents.train import learner, Learner
from tf_agents.train.triggers import PolicySavedModelTrigger
from tf_agents.train.utils import spec_utils, train_utils
from tfagents_system.train_params_base import TrainParamsBase
from tfagents_system.utils.comp import config_tf
import time
from utilities import setup_logging, import_params_module, check_training_done


osp = os.path


def train(exp_dir, ps: TrainParamsBase, reverb_port, logger: logging.Logger):
  # exit if training has finished
  train_step = train_utils.create_train_step()
  if check_training_done(exp_dir, train_step, ps):
    logger.info('Training done')
    return

  replay_buffer_ckpt_filename = osp.join(exp_dir, 'replay_buffer_ckpt_path.txt')
  args, kwargs = ps.collect_env_args_kwargs()
  env = ps.env_ctor(*args, **kwargs)
  obs_spec, action_spec, time_step_spec = spec_utils.get_tensor_specs(env)
  
  agent = ps.agent_ctor(train_step, obs_spec, action_spec, time_step_spec)

  ckpt_interval = ps.num_steps_per_iter * ps.checkpoint_period_itrs
  save_model_trigger = PolicySavedModelTrigger(osp.join(exp_dir, learner.POLICY_SAVED_MODEL_DIR), agent, train_step,
                                               interval=ckpt_interval, save_greedy_policy=True)

  variables = {
    reverb_variable_container.POLICY_KEY: agent.collect_policy.variables(),
    reverb_variable_container.TRAIN_STEP_KEY: train_step
  }
  # container holding policy weights
  reverb_address = f'localhost:{reverb_port}'
  reverb_client = reverb.Client(reverb_address)  # for checkpointing
  variable_container = ReverbVariableContainer(reverb_address, table_names=[reverb_variable_container.DEFAULT_TABLE])
  
  # number of consecutive transitions to sample from replay buffer
  sequence_length = ps.recurrent_sequence_length if ps.recurrent else 2
  replay_buffer = ReverbReplayBuffer(agent.collect_data_spec, sequence_length=sequence_length,
                                     table_name=reverb_replay_buffer.DEFAULT_TABLE, server_address=reverb_address,
                                     dataset_buffer_size=ps.prefetch_batches*ps.train_batch_size,
                                     num_workers_per_iterator=int(os.environ['OMP_NUM_THREADS']))
  
  # Function to initialize the dataset.
  def experience_dataset_fn():
    def _filter_invalid_transition(trajectories, unused_arg1):
      return tf.reduce_all(~trajectories.is_boundary()[:-1])
    dataset = replay_buffer.as_dataset(sample_batch_size=ps.train_batch_size, num_steps=sequence_length)
    dataset = dataset.unbatch().filter(_filter_invalid_transition).batch(ps.train_batch_size)
    dataset = dataset.prefetch(ps.prefetch_batches)
    return dataset

  mylearner = Learner(exp_dir, train_step, agent, experience_dataset_fn, triggers=[save_model_trigger, ],
                      checkpoint_interval=ckpt_interval, summary_interval=ps.num_steps_per_iter*ps.log_period_itrs,
                      max_checkpoints_to_keep=2)
  mylearner.train_summary_writer.set_as_default()
  variable_container.push(variables)

  max_train_steps = ps.num_iter * ps.num_steps_per_iter
  while train_step.numpy() < max_train_steps:
    itr = train_step.numpy() // ps.num_steps_per_iter
    logger.info(f'training iteration {itr+1} / {ps.num_iter}')

    start_time = time.time()
    losses: LossInfo = mylearner.run(iterations=ps.num_steps_per_iter)
    elapsed_time = time.time() - start_time
    logger.debug(f'waiting to push policy, itr {itr}')
    variable_container.push(variables)
    logger.debug(f'pushed policy, itr {itr}')
    
    if itr % ps.log_period_itrs == 0:
      tf.summary.scalar(name='Metrics/TrainSpeed', data=ps.num_steps_per_iter/elapsed_time, step=train_step)

    if itr % ps.checkpoint_period_itrs == 0:
      ckpt_path: str = reverb_client.checkpoint()
      with open(replay_buffer_ckpt_filename, 'w') as f:
        f.writelines([ckpt_path, ])
  
  logger.info('Training done')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-p", "--params", type=str, default="tfagents_system/default_params.py", help="params file")
  parser.add_argument('--reverb_port', type=int, default=8008)
  parser.add_argument("-c", "--config", type=str, default="config/simple.json", help="config file")
  parser.add_argument("-o", "--output", type=str, default="output", help="output folder, MUST exist")
  args = parser.parse_args()
  config_tf(1, int(os.environ['OMP_NUM_THREADS']))
  # tf.config.run_functions_eagerly(True)
  log_filename = osp.join(args.output, 'training_log.txt')
  logger = setup_logging(filename=log_filename, level=logging.INFO, name='train')

  exp_config_filename = osp.join(args.output, 'config.json')
  exp_params_filename = osp.join(args.output, 'params.py')
  if osp.isfile(exp_config_filename):  # resuming
    logger.warning(f'using {exp_config_filename} instad of {args.config}')
    logger.warning(f'using {exp_params_filename} instead {args.params}')
  else:  # starting from scratch
    copyfile(args.config, exp_config_filename)
    copyfile(args.params, exp_params_filename)

  params_module = import_params_module(exp_params_filename)
  with open(exp_config_filename, 'r') as f:
    config = json.load(f)
  ps: TrainParamsBase = params_module.TrainParams(config)

  train(args.output, ps, args.reverb_port, logger)