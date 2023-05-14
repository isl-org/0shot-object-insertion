"""
Downward TF Policy
"""
import numpy as np
from tf_policy_actionserver_base import TFPolicyActionServerBase
from tf_agents.specs.array_spec import sample_spec_nest
from tf_agents.trajectories import TimeStep, PolicyStep
from tf_agents.policies.py_policy import PyPolicy
import transforms3d.axangles as txa
import transforms3d.quaternions as txq


class RandomSearchPyPolicy(PyPolicy):
  def __init__(self, time_step_spec, action_spec, policy_state_spec=(), info_spec=(),
               observation_and_action_constraint_splitter=None):
    self.target_q = np.array([0.0, 1.0, 0.0, 0.0])
    self.counters = {'up': -1, 'down': 0, 'horizontal': -1}
    self.force_limit = 10.0  # N
    self.force_bias = None
    self.horizontal_vector = np.zeros(3)
    self.horizontal_limits = [[-0.1, -0.1, 0.0], [0.1, 0.1, 0.0]]
    self.n_horizontal_steps = 30
    super().__init__(time_step_spec, action_spec, policy_state_spec, info_spec,
                     observation_and_action_constraint_splitter)

  
  def _action(self, time_step: TimeStep, policy_state=(), seed=None) -> PolicyStep:
    obs = time_step.observation['obs'][-1]
    if self.force_bias is None:
      self.force_bias = np.copy(obs[:3])
    force = np.copy(obs[:3]) - self.force_bias
    
    # keep end effector pointing straight down
    a = obs[-3:]
    angle = np.linalg.norm(a)
    if abs(angle) < 1e-3:
      vector = np.array([1.0, 0.0, 0.0])
    else:
      vector = a / angle
    q_current = txq.axangle2quat(vector, angle)
    # if self.target_q is None:
    #   self.target_q = np.copy(q_current)
    q_target = txq.qmult(self.target_q, txq.qconjugate(q_current))
    vector, angle = txa.mat2axangle(txq.quat2mat(q_target))
    action = np.zeros(self.action_spec.shape)
    action[3:6] = angle * vector

    if self.counters['down'] >= 0:
      if force[2] > -self.force_limit:
        action[:3] = [0.0, 0.0, -0.2]
      else:
        self.counters['down'] = -1
        self.counters['up'] = 0
    elif self.counters['up'] >= 0:
      if force[2] > 0.5:
        self.counters['up'] = -1
        self.counters['horizontal'] = 0
      else:
        action[:3] = [0.0, 0.0, 0.2]
    elif self.counters['horizontal'] >= 0:
      if self.counters['horizontal'] == 0:
        self.horizontal_vector = np.random.uniform(*self.horizontal_limits)
      action[:3] = np.copy(self.horizontal_vector)
      self.counters['horizontal'] += 1
      if self.counters['horizontal'] > self.n_horizontal_steps:
        self.counters['horizontal'] = -1
        self.counters['down'] = 0
    else:
      raise ValueError('none of the counters are active')

    action = action.astype(self.action_spec.dtype)
    return PolicyStep(action, policy_state, ())


class RandomSearchPolicyActionServer(TFPolicyActionServerBase):
  def __init__(self, service_name='random_search_policy', **kwargs):
    super(RandomSearchPolicyActionServer, self).__init__(service_name, do_init=False, **kwargs)
    self.config['base_weight'] = 0.0
    self.config['residual_weight'] = 1.0
    self.init()
    self.policy = RandomSearchPyPolicy(self.time_step_spec, self.action_spec)
    self.policy_state = self.policy.get_initial_state()
    self.ts_template: TimeStep = sample_spec_nest(self.policy.time_step_spec, np.random)


if __name__ == '__main__':
  goal = {
    'observation': {
      'header': {
        'stamp': {
          'secs': 0,
          'nsecs': 0
        }
      },
      'g_T_ee': {
        'position': {'x': 0.0, 'y': 0.0, 'z': 0.5},
        'orientation': {'w': 1.0, 'x': 0.0, 'y': 0.0, 'z': 0.0}
      },
      'wrench': {
        'force': {'x': 0.0, 'y': 0.0, 'z': 0.0},
        'torque': {'x': 0.0, 'y': 0.0, 'z': 0.0},
      }
    },
  }
  rsp = RandomSearchPolicyActionServer()
  a = rsp.execute_cb(goal)