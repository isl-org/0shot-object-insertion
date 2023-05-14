"""
all-0 policy but goal follows a circular trajectory
"""
import numpy as np
from tf_agents.policies.py_policy import PyPolicy
from tf_agents.specs.array_spec import sample_spec_nest
from tf_agents.trajectories import TimeStep, PolicyStep
from tf_policy_actionserver_base import TFPolicyActionServerBase
import rospy


class PointTrackerPyPolicy(PyPolicy):
  def __init__(self, time_step_spec: TimeStep, action_spec, max_pos_action, max_rot_action, **kwargs):
    super(PointTrackerPyPolicy, self).__init__(time_step_spec, action_spec, **kwargs)
    self.max_pos_action = max_pos_action
    self.max_rot_action = max_rot_action

  def _action(self, time_step: TimeStep, policy_state=(), seed=None) -> PolicyStep:
    action = np.zeros(self.action_spec.shape)
    pos, rot = np.split(np.copy(time_step.observation['obs'][-1, 6:]), 2)
    action[:3] = pos / self.max_pos_action
    action[3:6] = rot / self.max_rot_action
    return PolicyStep(action, policy_state, ())


class CirclePolicyActionServer(TFPolicyActionServerBase):
  def __init__(self, service_name='circle_policy', **kwargs):
    super(CirclePolicyActionServer, self).__init__(service_name, do_init=False, **kwargs)
    self.circle_r = 0.05
    self.circle_time = 5.0
    self.circle_c = np.zeros(3)
    self.theta = np.linspace(0, 2*np.pi, 500)
    self.config['base_weight'] = 0.0
    self.config['residual_weight'] = 1.0
    self.init()
    self.policy = PointTrackerPyPolicy(self.time_step_spec, self.action_spec, self.config['max_pos_action'],
                                       self.config['max_rot_action'])
    self.ts_template: TimeStep = sample_spec_nest(self.policy.time_step_spec, np.random)

  def _goal2ts(self, goal: dict) -> TimeStep:
    super(CirclePolicyActionServer, self)._goal2ts(goal)
    stamp = goal['observation']['header']['stamp']
    time = rospy.Time(stamp['secs'], stamp['nsecs']).to_sec()
    angle = np.pi * (1.0 - np.cos(2 * np.pi * time / self.circle_time))
    p = np.array([self.circle_r*np.cos(angle), self.circle_r*np.sin(angle), 0.0])
    ts: TimeStep = self.ts_template
    ts.observation['obs'][-1, 6:9] += self.circle_c + p


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
  cp = CirclePolicyActionServer()
  cp.execute_cb(goal)