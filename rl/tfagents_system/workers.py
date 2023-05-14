from collections import deque
import logging
import numpy as np
import os.path as osp
import tensorflow as tf
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.experimental.distributed.reverb_variable_container import ReverbVariableContainer
from tf_agents.policies.random_py_policy import RandomPyPolicy
from tf_agents.train.actor import Actor
from tf_agents.train import learner
from tf_agents.train.utils import train_utils
from tfagents_system.utils.metric import compute
from tfagents_system.train_params_base import TrainParamsBase
import time
from utilities import setup_logging, RunningMean


class Worker(object):
  def __init__(self, id, exp_dir, ps: TrainParamsBase, reverb_port, seeds):
    setup_logging(level=logging.INFO)
    self.id = id
    self.logger = setup_logging(level=logging.INFO, name=f'collect_worker_{self.id}')
    self.collect_policy = train_utils.wait_for_policy(
      osp.join(exp_dir, learner.POLICY_SAVED_MODEL_DIR, learner.COLLECT_POLICY_SAVED_MODEL_DIR),
      load_specs_from_pbtxt=True)
    self.greedy_policy = train_utils.wait_for_policy(
      osp.join(exp_dir, learner.POLICY_SAVED_MODEL_DIR, learner.GREEDY_POLICY_SAVED_MODEL_DIR),
      load_specs_from_pbtxt=True)
    self.ps = ps
    self.reverb_port = reverb_port
    self.seeds = seeds
    self.n_workers = self.ps.compute_config['n_parallel_workers']
    
    self.train_step = train_utils.create_train_step()
    self.collect_policy_variables = {
      reverb_variable_container.POLICY_KEY: self.collect_policy.variables(),
      reverb_variable_container.TRAIN_STEP_KEY: self.train_step
    }
    self.greedy_policy_variables = {
      reverb_variable_container.POLICY_KEY: self.greedy_policy.variables(),
      reverb_variable_container.TRAIN_STEP_KEY: self.train_step
    }
    self.variable_container = ReverbVariableContainer(f'localhost:{reverb_port}',
                                                      table_names=[reverb_variable_container.DEFAULT_TABLE])
    self.pull_policy()

    args, kwargs = ps.collect_env_args_kwargs(id=self.id)
    self.collect_env: PyEnvironment = ps.env_ctor(seed=self.seeds[0], *args, **kwargs)
    args, kwargs = ps.eval_env_args_kwargs(id=self.id, collect_env=self.collect_env)
    self.eval_env: PyEnvironment = ps.env_ctor(seed=self.seeds[1], *args, **kwargs)
    self.collect_observers = self.ps.train_observers_ctor(self.reverb_port, collect_only=False)
    self.eval_observers = self.ps.eval_observers_ctor()
    self.n_collect_steps = self._per_worker(self.ps.n_train_episodes_per_iter*self.ps.env_config['horizon'])
    summary_dir = osp.join(exp_dir, learner.TRAIN_DIR, f'{id}')
    self.collect_actor = Actor(self.collect_env, self.collect_policy, self.train_step,
                               steps_per_run=self.n_collect_steps,
                               observers=self.collect_observers[0].values(),
                               transition_observers=self.collect_observers[1].values(),
                               metrics=list(self.collect_observers[2].values()),
                               summary_dir=summary_dir,
                               summary_interval=self.ps.num_steps_per_iter*self.ps.log_period_itrs)
    self.collect_actor.metrics.extend(self.collect_observers[1].values())
    summary_dir = osp.join(exp_dir, 'eval', f'{id}')
    self.eval_summary_writer = tf.summary.create_file_writer(summary_dir, flush_millis=10*1000)
    self.collect_speed = RunningMean()
    self.eval_speed = RunningMean()

  
  def run(self):
    max_train_steps = self.ps.num_iter * self.ps.num_steps_per_iter
    curriculum = {}
    for k,v in self.ps.env_config.get('curriculum', {}).items():
      curriculum[k] = {
        'levels': deque(np.linspace(*v['range'], len(v['iters'])).astype(type(v['range'][0]))),
        'iters':  deque(np.sort(v['iters']))
      }
    
    while self.train_step.numpy() < max_train_steps:
      itr = self.train_step.numpy() // self.ps.num_steps_per_iter
      
      this_params = {}
      for k,v in curriculum.items():
        while len(v['iters']) and (itr >= v['iters'][0]):
          v['iters'].popleft()
          this_params[k] = v['levels'].popleft()
          self.logger.info(f'itr {itr} {k}: {this_params[k]}')
      if this_params:
        self.collect_env.set_state(this_params)
        self.collect_actor.reset()

      self.logger.debug(f'itr {itr} waiting to pull policy')
      self.pull_policy()
      self.logger.debug(f'itr {itr} pulled policy')
      self.logger.debug(f'itr {itr} collecting...')
      self.collect(itr%self.ps.log_period_itrs == 0)
      self.logger.debug(f'itr {itr} collection done')
      if (self.ps.eval_period_itrs > 0) and (itr % self.ps.eval_period_itrs == 0):
        self.eval()


  def collect_random(self):
    random_policy = RandomPyPolicy(self.collect_env.time_step_spec(), self.collect_env.action_spec())
    n_steps = self._per_worker(2 * self.ps.n_buffer_init_episodes * self.ps.env_config['horizon'])
    observers, transition_observers, metrics = self.ps.train_observers_ctor(self.reverb_port, collect_only=False)
    
    actor = Actor(self.collect_env, random_policy, self.train_step, steps_per_run=n_steps, observers=observers.values(),
                  transition_observers=transition_observers.values(), metrics=list(metrics.values()))
    actor.metrics.extend(transition_observers.values())
    
    count = 0
    steps_written = 0.
    while (count < 10) and (steps_written < n_steps):
      actor.run()
      steps_written += observers['steps_written'].result()
      observers['steps_written'].reset()
      count += 1
    actor.log_metrics()
    
    observers['replay_buffer'].close()  # close() includes flush()
    self.collect_env.reset()


  def collect(self, do_logging: bool):
    start_time = time.time()
    count = 0
    steps_written = 0.
    while (count < 1) and (steps_written < self.n_collect_steps):
      self.collect_actor.run()
      steps_written += self.collect_observers[0]['steps_written'].result()
      self.collect_observers[0]['steps_written'].reset()
      count += 1
    elapsed_time = time.time() - start_time
    self.collect_actor.log_metrics()
    
    self.collect_speed.update(self.collect_actor._driver._max_steps / elapsed_time)
    self.collect_observers[0]['replay_buffer'].flush()
    
    if do_logging:
      with self.collect_actor._summary_writer.as_default():
        tf.summary.scalar(name='Metrics/CollectSpeed', data=self.collect_speed.get_value(), step=self.train_step)
      self.collect_speed.reset()


  def eval(self):
    o = self.eval_observers[2]
    # last observer has to be env_steps, because compute() will reset only the last observer
    observers = list(self.eval_observers[0].values()) + [o['reward'], o['episode_length'], o['env_steps']]
    start_time = time.time()
    compute(observers, self.eval_observers[1].values(), self.eval_env, self.greedy_policy,
            num_episodes=self.ps.n_eval_episodes)
    elapsed_time = time.time() - start_time
    self.eval_speed.update(self.eval_observers[2]['env_steps'].result() / elapsed_time)
    d = {**self.eval_observers[0], **self.eval_observers[1], **self.eval_observers[2]}
    with self.eval_summary_writer.as_default():
      for metric in d.values():
        tf.summary.scalar(name=f'Metrics/{metric.name}', data=metric.result(), step=self.train_step)
      tf.summary.scalar(name='Metrics/EvalSpeed', data=self.eval_speed.get_value(), step=self.train_step)
      self.eval_speed.reset()


  def close(self):
    self.collect_observers[0]['replay_buffer'].close()
    self.eval_env.close()
    self.collect_env.close()
    self.eval_summary_writer.flush()


  def pull_policy(self):
    self.variable_container.update(self.collect_policy_variables)
    self.variable_container._assign(self.greedy_policy_variables, self.collect_policy_variables)

  
  def _per_worker(self, x: int):
    return (x + self.n_workers-1) // self.n_workers
