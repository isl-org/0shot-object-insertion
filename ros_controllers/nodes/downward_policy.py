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


class DownwardPyPolicy(PyPolicy):
  def __init__(self, time_step_spec, action_spec, policy_state_spec=(), info_spec=(),
               observation_and_action_constraint_splitter=None):
    self.target_q = np.array([0.0, 1.0, 0.0, 0.0])
    super().__init__(time_step_spec, action_spec, policy_state_spec, info_spec,
                     observation_and_action_constraint_splitter)

  def _action(self, time_step: TimeStep, policy_state=(), seed=None) -> PolicyStep:
    obs = time_step.observation['obs']
    a = obs[0, -3:]
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
    action[2] = -0.2
    action[3:6] = angle * vector
    action = action.astype(self.action_spec.dtype)

    return PolicyStep(action, policy_state, ())


class DownwardPolicyActionServer(TFPolicyActionServerBase):
  def __init__(self, service_name='downward_policy', **kwargs):
    super(DownwardPolicyActionServer, self).__init__(service_name, do_init=False, **kwargs)
    self.config['base_weight'] = 0.0
    self.config['residual_weight'] = 1.0
    self.init()
    self.policy = DownwardPyPolicy(self.time_step_spec, self.action_spec)
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
  dp = DownwardPolicyActionServer()
  a = dp.execute_cb(goal)