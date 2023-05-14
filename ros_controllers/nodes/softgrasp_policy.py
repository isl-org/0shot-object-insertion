import argparse
import json
import numpy as np
import os.path as osp
import rospy
from tf_agents.policies import policy_loader
from tf_agents.specs.array_spec import sample_spec_nest
from tf_agents.trajectories import TimeStep
from tf_policy_actionserver_base import TFPolicyActionServerBase
from stable_baselines3 import PPO

class SoftgraspPolicyActionServer(TFPolicyActionServerBase):
  def __init__(self, exp_dir, checkpoint_dirname, service_name='softgrasp_policy', do_init=True, **kwargs):
    super(SoftgraspPolicyActionServer, self).__init__(service_name, do_init=False, **kwargs)
    
    exp_dir = osp.expanduser(exp_dir)
    with open(osp.join(exp_dir, 'config.json'), 'r') as f:
      self.config = json.load(f)
    self.config['max_rot_action'] = np.deg2rad(self.config['max_rot_action_deg'])
    
    # load policy
    policy_dir = osp.join(exp_dir, 'policies', 'greedy_policy')
    ckpt_dir = osp.join(exp_dir, 'policies', 'checkpoints', checkpoint_dirname)
    rospy.loginfo(f'Loading policy from {policy_dir} and {ckpt_dir}')
    self.policy = policy_loader.load(policy_dir, ckpt_dir)
    self.policy_state = self.policy.get_initial_state(1)
    self._action_spec = self.policy.action_spec
    self._time_step_spec = self.policy.time_step_spec
    self.ts_template: TimeStep = sample_spec_nest(self._time_step_spec, np.random)

    if self.ppo:
      filename = "/home/sbrahmbh/softgrasp_ws/src/softgrasp_ros_controllers/policies/exp_ppo/checkpoint_5780000.zip"
      self.ppo_policy = PPO.load(filename)
    
    if do_init:
      self.init()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--exp_dir', required=True)
  parser.add_argument('--ckpt_dirname', required=True)
  args = parser.parse_args()
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
  cp = SoftgraspPolicyActionServer(args.exp_dir, args.ckpt_dirname)
  cp.execute_cb(goal)