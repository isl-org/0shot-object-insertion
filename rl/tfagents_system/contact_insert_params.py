from tf_agents.environments.py_environment import PyEnvironment
from envs.contact_insert_py_environment import ContactInsertPyHistoryEnv
from tfagents_system.utils.metric import SuccessRateMetric
from tfagents_system.train_params_base import TrainParamsBase
import tensorflow as tf


class TrainParams(TrainParamsBase):
  def __init__(self, env_config: dict):
    super(TrainParams, self).__init__(env_config)
    self.has_renderer = False
    self.num_iter = 128000
    self.n_episodes_in_buffer = 8000
    self.n_buffer_init_episodes = 500
    if self.env_config['include_actions_in_history']:
      ppc_fn = lambda x: tf.concat((x['obs']['observation'], x['obs']['action']), axis=-1)
    else:
      ppc_fn = lambda x: x['obs']
    self.preprocessing_combiner = tf.keras.layers.Lambda(ppc_fn)

  
  def env_ctor(self, seed=None, *args, **kwargs) -> PyEnvironment:
    env = ContactInsertPyHistoryEnv(has_renderer=self.has_renderer, config=self.env_config, discount=self.discount,
                                    seed=seed, *args, **kwargs)
    return env

  
  def collect_env_args_kwargs(self, *args, **kwargs):
    return (), {'eval_mode': False, 'noise_level': 0.0}
  
  
  def eval_env_args_kwargs(self, *args, **kwargs):
    # forced_init_idx = 0 because we always want to evaluate with the EEF starting at a "high" pose
    # state_idx will not be used because sequential_eval == False, but we set it for completeness
    collect_env: ContactInsertPyHistoryEnv = kwargs['collect_env']
    state_idx = kwargs['id'] % collect_env.get_n_eval_states()
    return (), {'eval_mode': True, 'forced_init_idx': 0, 'state_idx': state_idx, 'noise_level': 1.0}
  
  
  def train_observers_ctor(self, reverb_port, collect_only=False):
    observers, transition_observers, metrics = super(TrainParams, self).train_observers_ctor(reverb_port, collect_only)
    buffer_size = self.log_period_itrs * self.n_train_episodes_per_iter
    transition_observers = dict(transition_observers, success_rate=SuccessRateMetric(buffer_size=buffer_size))
    return observers, transition_observers, metrics
  

  def eval_observers_ctor(self):
    observers, transition_observers, metrics = super(TrainParams, self).eval_observers_ctor()
    transition_observers = dict(transition_observers, success_rate=SuccessRateMetric(buffer_size=self.n_eval_episodes))
    return observers, transition_observers, metrics