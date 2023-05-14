from envs.contact_insert_environment import ContactInsertEnv
import numpy as np
from robosuite.wrappers.gym_wrapper import GymWrapper
from tf_agents.environments.wrappers import HistoryWrapper
from tf_agents.specs import ArraySpec, BoundedArraySpec
from tf_agents.trajectories import TimeStep
from tf_agents.trajectories import time_step as ts
from tf_agents.environments.py_environment import PyEnvironment


class ContactInsertPyEnv(PyEnvironment):
    def __init__(self, config: dict, discount=1.0, **env_kwargs):
      env = ContactInsertEnv(**config, **env_kwargs)
      self.obs_keys = env.obs_keys
      self._env = GymWrapper(env, keys=env.obs_keys)
      self._discount = discount
      self.get_specs()
      self._time_step = None
      self._rendering = None
      self._episode_ended = False

    
    def get_specs(self):
      self._observation_spec = {'obs': ArraySpec(shape=(self._env.obs_dim,), dtype=np.float32, name='observation'),
                                'aux': {'successful': ArraySpec(shape=(), dtype=np.int32, name='auxiliary')}}
      self._action_spec = BoundedArraySpec(shape=(self._env.action_dim,), dtype=np.float32, minimum=-1, maximum=1,
                                           name='action')


    def action_spec(self):
      return self._action_spec


    def observation_spec(self):
      return self._observation_spec


    def render(self, mode):
      return self._rendering


    def _reset(self) -> ts.TimeStep:
      obs = self._env.reset()
      self._episode_ended = False
      return ts.restart({'obs': obs.astype(np.float32), 'aux': {'successful': np.array(1, dtype=np.int32)}})


    def _step(self, action) -> ts.TimeStep:
      if self._episode_ended:
        return self.reset()
      obs, reward, self._episode_ended, info = self._env.step(action)
      obs = {'obs': obs.astype(np.float32), 'aux': {'successful': np.array(info['done_reason']==2, dtype=np.int32)}}
      time_step = ts.termination(obs, reward) if self._episode_ended else ts.transition(obs, reward, self._discount)
      return time_step


    def start_recording(self, video_path):
      self._env.start_recording(video_path)


    def end_recording(self):
      self._env.end_recording

    
    def set_state(self, state_dict: dict) -> None:
      """
      change some attributes of the env. Expects a dict whose keys are attribute names and values are new attribute
      values
      raises KeyError if attribute is not found
      """
      self._env.set_state(state_dict)


    def get_state(self) -> dict:
      return self._env.get_state()

    
    def get_n_eval_states(self):
      return self._env.n_eval_states


class ContactInsertPyHistoryEnv(HistoryWrapper):
    def __init__(self, config: dict, **env_kwargs):
      self.config = config
      self._env = ContactInsertPyEnv(config=config, **env_kwargs)
      super(ContactInsertPyHistoryEnv, self).__init__(self._env,
                                                    history_length = config['history_length'],
                                                    include_actions = config['include_actions_in_history'],
                                                    tile_first_step_obs = True)
      if self.config['include_actions_in_history']:
        self._observation_spec['obs'] = {'observation': self._observation_spec['observation']['obs'],
                                         'action': self._observation_spec['action']
                                        }
        self._observation_spec.pop('observation')
        self._observation_spec.pop('action')
      self._observation_spec['aux'] = {'successful': ArraySpec(shape=(), dtype=np.int32, name='auxiliary')}

    def _modify_ts(self, in_ts: ts.TimeStep):
      if self.config['include_actions_in_history']:
        new_ts = TimeStep(in_ts.step_type,
                          in_ts.reward,
                          in_ts.discount,
                          {'obs': {'observation': in_ts.observation['observation']['obs'],
                                   'action': in_ts.observation['action']},
                            'aux': {'successful': in_ts.observation['observation']['aux']['successful'][-1]}
                          })
      else:
        new_ts = TimeStep(in_ts.step_type,
                          in_ts.reward,
                          in_ts.discount,
                          {'obs': in_ts.observation['obs'],
                           'aux': {'successful': in_ts.observation['aux']['successful'][-1]}
                          })
      return new_ts


    def _reset(self) -> ts.TimeStep:
      original_ts = super(ContactInsertPyHistoryEnv, self)._reset()
      new_ts = self._modify_ts(original_ts)
      return new_ts


    def _step(self, action) -> ts.TimeStep:
      is_last = self._current_time_step.is_last()
      original_ts = super(ContactInsertPyHistoryEnv, self)._step(action)
      new_ts = original_ts if is_last else self._modify_ts(original_ts) # modify only if reset hasn't been called. otherwise
                                                                        # it's already modified
      return new_ts

    
    def start_recording(self, video_path):
      self._env._env.start_recording(video_path)

    
    def end_recording(self):
      self._env._env.end_recording()


    def set_state(self, state_dict: dict) -> None:
      """
      change some attributes of the env. Expects a dict whose keys are attribute names and values are new attribute
      values
      raises KeyError if attribute is not found
      """
      self._env.set_state(state_dict)


    def get_state(self) -> dict:
      return self._env.get_state()

    
    def get_n_eval_states(self):
      return self._env.get_n_eval_states()