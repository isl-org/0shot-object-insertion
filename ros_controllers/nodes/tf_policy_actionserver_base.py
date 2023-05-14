from collections import deque
import math
import numpy as np
from roslibpy import Ros
import roslibpy
from roslibpy.actionlib import SimpleActionServer
from tf_agents.policies.py_policy import PyPolicy
from tf_agents.specs.array_spec import ArraySpec, BoundedArraySpec
from tf_agents.trajectories import TimeStep
import transform_utils as T


class TFPolicyActionServerBase(object):
  def __init__(self, service_name='policy', host='localhost', port=9090, seed=None, do_init=True, ppo=False,
               include_velocity=False):
    self.ppo = ppo
    self.include_velocity = include_velocity
    self._ros = Ros(host=host, port=port)
    self._as = SimpleActionServer(self._ros, service_name, 'softgrasp_ros_controllers/TFPolicyAction')
    self._telemetry_pub = roslibpy.Topic(self._ros, 'telemetry', 'softgrasp_ros_controllers/Telemetry')
    self.policy: PyPolicy = None
    self.ppo_policy = None
    self.policy_state = ()
    self.past_goal_pose = np.zeros(6)
    self.ts_template: TimeStep = None
    self._action = np.zeros(7, dtype=np.float64)
    self.config = {
        'max_pos_action': 0.5,
        'max_rot_action': np.deg2rad(45.0),
        'base_weight': 0.5,
        'residual_weight': 0.5,
        'history_length': 1,
    }
    self.obs_q = None
    self.base_pos_clip = None
    self.base_rot_clip = None
    self._action_spec = None
    self._time_step_spec = None
    # scaling information
    self.scaling_input_min = -np.ones(6)
    self.scaling_input_max =  np.ones(6)
    self.scaling_output_min = np.array([-0.15, -0.15, -0.15, -0.5, -0.5, -0.5])
    self.scaling_output_max = np.array([ 0.15,  0.15,  0.15,  0.5,  0.5,  0.5])
    self.action_scale = None
    self.action_output_transform = None
    self.action_input_transform = None
    if do_init:
      self.init()

  def init(self):
    w = self.config['base_weight']
    self.base_pos_clip = w*self.config['max_pos_action']
    self.base_rot_clip = w*self.config['max_rot_action']
    self.action_scale = abs(self.scaling_output_max - self.scaling_output_min) / abs(self.scaling_input_max - self.scaling_input_min)
    self.action_output_transform = (self.scaling_output_max + self.scaling_output_min) / 2.0
    self.action_input_transform = (self.scaling_input_max + self.scaling_input_min) / 2.0
    self.obs_q = deque([], maxlen=self.config['history_length'])
    self._telemetry_pub.advertise()

  @property
  def action_spec(self):
    if self._action_spec is None:
      self._action_spec = BoundedArraySpec(shape=(7, ), dtype=np.dtype('float32'), name='action', minimum=-1.0,
                                           maximum=1.0)
    return self._action_spec

  @property
  def time_step_spec(self):
    if self._time_step_spec is None:
      self._time_step_spec = TimeStep(
        discount=BoundedArraySpec(shape=(), dtype=np.dtype('float32'), name='discount', minimum=0.0, maximum=1.0),
        observation={
          'aux': ArraySpec(shape=(), dtype=np.dtype('int32'), name='auxiliary'),
          'obs': ArraySpec(shape=(self.config['history_length'], 12), dtype=np.dtype('float32'), name='observation')
        },
        reward=ArraySpec(shape=(), dtype=np.dtype('float32'), name='reward'),
        step_type=ArraySpec(shape=(), dtype=np.dtype('int32'), name='step_type')
      )
    return self._time_step_spec

  def _scale_residual_action(self, rpos, rrot):
    w = self.config['residual_weight']
    return w*rpos*self.config['max_pos_action'], w*rrot*self.config['max_rot_action']

  def _goal2ts(self, goal: dict) -> TimeStep:
    g_T_ee = goal['observation']['g_T_ee']
    pos = np.array([
      g_T_ee['position']['x'],
      g_T_ee['position']['y'],
      g_T_ee['position']['z'],
    ]).astype(np.float32)
    quat = np.array([
      g_T_ee['orientation']['x'],
      g_T_ee['orientation']['y'],
      g_T_ee['orientation']['z'],
      g_T_ee['orientation']['w'],
    ]).astype(np.float32)
    aa = T.quat2axangle(T.normalize_quat(quat))
    wrench = goal['observation']['wrench']
    force = np.array([
      wrench['force']['x'],
      wrench['force']['y'],
      wrench['force']['z'],
    ]).astype(np.float32)
    torque = np.array([
      wrench['torque']['x'],
      wrench['torque']['y'],
      wrench['torque']['z'],
    ]).astype(np.float32)
    # # for peg hole
    # force /= -5.0
    # torque /= -5.0
    if force[2] < 0.0:  # collision
      force = 15.0 * force
      torque = 7.5 * torque
    # goal velocity
    if self.include_velocity:
      goal_vel = self.past_goal_pose - np.hstack((pos, aa))
      self.past_goal_pose = np.hstack((pos, aa))

    N = 1 if len(self.obs_q) else self.config['history_length']
    for _ in range(N):
      if self.include_velocity:
        self.obs_q.append(np.hstack((force, torque, pos, aa, goal_vel)))
      else:
        self.obs_q.append(np.hstack((force, torque, pos, aa)))

    # fill ts observation
    self.ts_template.observation['obs'][...] = np.vstack(self.obs_q)

  def execute_cb(self, goal: dict) -> dict:
    """
    goal: softgrasp_ros_controllers::TFPolicyGoal
    return: softgrasp_ros_controllers::TFPolicyResult
    """
    # construct policy timestep
    self._goal2ts(goal)
    ts = self.ts_template

    # call policy
    if self.ppo:
      policy_action, _ = self.ppo_policy.predict(ts.observation['obs'], deterministic=True)
    else:
      policy_step = self.policy.action(ts, self.policy_state)
      policy_action = policy_step.action
      self.policy_state = policy_step.state
    # residual action
    rpos, rrot = self._scale_residual_action(policy_action[:3], policy_action[3:6])

    # base action
    bpos = T.clip_translation(ts.observation['obs'][-1, 6:9], self.base_pos_clip)
    brot = T.clip_axangle_rotation(ts.observation['obs'][-1, 9:12], self.base_rot_clip)
    
    # combine residual and base actions
    t = rpos + bpos
    q = T.axangle2quat(rrot)
    q = T.quat_multiply(T.axangle2quat(brot), q)
    q *= math.copysign(1, q[3])
    a = T.quat2axangle(q)

    # apply scaling
    action = np.clip(np.hstack((t, a)), self.scaling_input_min, self.scaling_input_max)
    action = (action - self.action_input_transform) * self.action_scale + self.action_output_transform
    
    self._action[:3] = action[:3]
    self._action[3:] = T.axangle2quat(action[3:])
    
    # construct response
    result = {'action': {}}
    a = result['action']
    a['header'] = goal['observation']['header']
    a['action'] = self._action.tolist()
    self._as.set_succeeded(result)

    # publish telemetry
    telemetry_msg = {
      'header': goal['observation']['header'],
      'observation': ts.observation['obs'].flatten().tolist(),
      'action': a['action']
    }
    self._telemetry_pub.publish(telemetry_msg)

  def run(self):
    self._as.start(self.execute_cb)
    try:
      self._ros.run_forever()
    except:
      print('Closed ROS connection')

  def reset(self):
    if self.obs_q:
      self.obs_q.clear()