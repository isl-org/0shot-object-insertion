from functools import partial
import glfw
import json
import logging
import math
import mujoco_py
import numpy as np
import pickle
import os.path as osp
import imageio
import copy

from robosuite import load_controller_config
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas.table_arena import TableArena
from robosuite.models.objects import MujocoXMLObject
from robosuite.models.objects.primitive import BoxObject
from robosuite.models.objects.generated_objects import CompositeObject
from robosuite.models.tasks import ManipulationTask
import robosuite.utils.macros as macros
from robosuite.utils.observables import Observable, sensor
import robosuite.utils.transform_utils as T


class ContactInsertEnv(SingleArmEnv):
  def __init__(
          self,
          robots='Panda',
          env_configuration="default",
          controller_configs=None,
          gripper_types="default",
          initialization_noise=None,
          table_full_size=(0.8, 0.8, 0.05),
          table_offset=(0, 0, 0.9),
          use_camera_obs=False,
          use_object_obs=True,
          has_renderer=False,
          has_offscreen_renderer=False,
          render_camera='frontview',
          render_collision_mesh=False,
          render_visual_mesh=True,
          render_gpu_device_id=-1,
          horizon=200,
          ignore_done=False,
          hard_reset=False,
          camera_names=None,
          camera_heights=0,
          camera_widths=0,
          camera_depths=False,
          init_grasp=False,
          init_paused=False,
          no_grasp_acquire=False,
          obs_keys=None,
          override_plate_pose=False,
          teleop=False,
          eval_mode=False,
          seed=None,
          sequential_eval=False,
          forced_init_idx=None,
          state_idx=0,
          noise_level=1.0,
          goalreach_steps=10,
          **config):
    """
    init_grasp: whether the first gripper action will be to open (False) or close (True)
    init_paused: whether the first viewer frame blocks. Effective only when has_renderer == True
    no_grasp_acquire: don't acquire grasp during reset. If config['init_state_filename'] is not None, don't check for
      collision with object in _acquire_grasp()
    obs_keys: list of observation names to be included in flattened obs vector. Used to create GymWrapper.
    override_plate_pose: Place plate at self.plate_{pos,rot}. Valid only when config['init_state_filename'] is not None
    eval_mode: Uses a deterministic set of pre-sampled EEF starting locations instead of live-sampling them
    sequential_eval: cycle sequentially through all the states in init_idx, rather than sampling randomly
    forced_init_idx: init_idx value that takes precedence over internal calculations. init_idx controls EEF or slot pose
      bins e.g. low medium high EEF, left middle right slot pose
    state_idx: controls which state will be chosen in the bin indicated by init_idx, used only if sequential_eval = True
    noise_level: attenuation factor in [0, 1] to be applied to all noise
    goalreach_steps: number of steps in goalreach state for episode to end
    """
    # World coordinate system: X out of screen, Y right, Z up
    # TODO: get point clouds of the destination and grasped object

    # setup time and frequencies
    macros.SIMULATION_TIMESTEP = 0.001 * config['time_scale']  # remember to also change in base.xml!
    control_freq = int(np.round(1.0 / (macros.SIMULATION_TIMESTEP * config['decision_step'])))

    self.table_full_size = np.array(table_full_size)
    self.table_friction = (1, 0.005, 0.0001)
    self.table_offset = table_offset
    self.use_object_obs = use_object_obs
    self.grasp_action = 1 if init_grasp else -1
    self.config = config
    self.base_pos_action = np.zeros(3)
    self.base_rot_action = np.zeros(3)
    self.slot_pos, self.slot_rot_quat = T.mat2pose(np.eye(4))
    self.done = False
    self.episode_started = False  # set when grasp has been acquired, reset at env reset()
    self.episode_done = False  # shadow flag to self.done, needed because self.done is manipulated by super class
    self.done_reason = -1  # -1: not done, 0: grasp lost, 1: time out, 2: goal reached
    self.obs_keys = obs_keys
    self.reward_scale = 1.0  # TODO
    self.pf = None
    self.logger = logging.getLogger(self.__class__.__name__)
    self.no_grasp_acquire = no_grasp_acquire
    self.override_plate_pose = override_plate_pose
    self.eval_mode = eval_mode
    self.sequential_eval = sequential_eval
    self.state_idx = state_idx
    self.init_idx = 0
    self.forced_init_idx = forced_init_idx
    self.noise_level = noise_level
    self.goalreach_steps = goalreach_steps
    self.rng: np.random.Generator = np.random.default_rng(seed)
    self.grasp_lost_counter = 0
    self.goal_reached_counter = 0
    # int: dist from central slot in slot size units. -/+ indicates left/right side of central slot
    self.closest_slot = None
    self.slot_base_plate_thickness = 0.02
    self.slot_base_plate_stability_extension = 0.06
    self.half_slot_breadth = 0.15
    self.past_external_wrench = np.zeros(6)

    # process and log the config
    if self.config['randomize_eef_location'] and self.config['randomize_goal_location']:
      self.logger.warning('cannot randomize both end-effector and goal. Turning off end-effector randomization')
      self.config['randomize_eef_location'] = False
    # assign higher probability to cases where eef starts high
    self.init_idx_p = [0.5, 0.25, 0.25] if self.config['randomize_goal_location'] else [1.0/3, 1.0/3, 1.0/3]
    self.config['residual_weight_bak'] = self.config['residual_weight']
    self.config['base_weight_bak'] = self.config['base_weight']
    self.config['max_rot_action'] = np.sqrt(3) * np.deg2rad(self.config['max_rot_action_deg'])
    self.config['success_rot_thresh'] = np.deg2rad(self.config['success_rot_thresh_deg'])
    self.config['goaldist_rot_penalty_cutoff'] = np.deg2rad(self.config['goaldist_rot_penalty_cutoff_deg'])
    
    # decide penalty magnitudes
    self.config['time_penalty'] = (1.0 / horizon) if self.config['use_time_penalty'] else 0.0
    self.goaldist_pos_penalty_slope = 0.5 * self.config['drop_penalty'] / \
        (self.config['goaldist_pos_penalty_cutoff'] * horizon)
    self.goaldist_rot_penalty_slope = 0.5 * self.config['drop_penalty'] / \
        (self.config['goaldist_rot_penalty_cutoff'] * horizon)
    if (self.goaldist_pos_penalty_slope*self.config['goaldist_pos_penalty_cutoff'] < 2*self.config['time_penalty']) or \
            (self.goaldist_rot_penalty_slope*self.config['goaldist_rot_penalty_cutoff'] < 2*self.config['time_penalty']):
      self.logger.warning('*** please increase drop penalty or decrease time penalty ***')
    self.config['force_penalty_mag'] = self.config['drop_penalty'] / (0.5 * horizon)
    self.config['torque_penalty_mag'] = self.config['drop_penalty'] / (0.5 * horizon)
    if self.config['force_penalty_mag'] < self.config['time_penalty']:
      self.logger.warning('*** please increase drop penalty or decrease time penalty ***')
    # self.logger.info('Config:')
    # for key, val in self.config.items():
    #   self.logger.info(f'{key}: {val}')
    if self.config['num_slots'] <= 0:
      raise ValueError('config num_slots must be positive')
    if self.config['num_slots'] % 2 != 1:
      raise ValueError('config num_slots must be odd')
    if np.min(self.config['policy_lag_range']) < 0:
      self.logger.warning('lag must be non-negative. Setting lag to 0')
      self.config['policy_lag_range'] = [0, 1]
    if not (self.config['policy_lag_range'][0]%1 == 0 and self.config['policy_lag_range'][1]%1 == 0):
      self.logger.warning('lag must be non-negative integer. Rounding to nearest integer')
      self.config['policy_lag_range'] = [int(val) for val in self.config['policy_lag_range']]

    object_kwargs = dict(solref=(0.001, 1), solimp=(0.998, 0.998, 0.001), friction=(0.95, 0.3, 0.1), density=2000.0)
    table_top = np.array([0, 0, table_offset[2]])
    table_hxy_size = 0.5*np.array([table_full_size[0], table_full_size[1], 0])
    
    # plate
    self.plate = MujocoXMLObject(osp.join('assets', 'objects', 'simple_plate.xml'), name='plate')

    # two cube supports that allow the plate to hang over the edge of the table, so that it can be grasped
    support_hsize = np.array([0.025, 0.025, 0.025])
    self.supports = [
      BoxObject('support1', support_hsize, **object_kwargs),
      BoxObject('support2', support_hsize, **object_kwargs),
    ]
    support_offset = np.array([0, support_hsize[1], 0])
    support_gap = np.array([0, 0.08, 0])
    self.supports_pos = [table_top - b.bottom_offset + table_hxy_size -
      (support_hsize[0], 0, 0) - (2*i+1)*support_offset - i*support_gap for i,b in enumerate(self.supports)]
    
    # position plate against the support
    self.plate_pos = np.mean(self.supports_pos, axis=0) - self.supports[0].bottom_offset - self.plate.bottom_offset

    self.slotset = self._construct_slot(**object_kwargs)
    if self.config['randomize_goal_location']:
      self.slot_pos_range = np.array(self.config['slot_pos_range'])
      self.slot_rot_range_deg = np.array(self.config['slot_rot_range_deg'])
    else:
      self.slot_pos_range = np.zeros((3, 2, 3))
      self.slot_rot_range_deg = np.zeros((3, 2))
    self.slot_pos_range += (table_top - self.slotset.bottom_offset)
    # add small vertical offset to prevent spurious collisions with tabletop
    self.slot_pos_range += np.array([0, 0, 1e-3])

    # pose of the object coordinate system goal w.r.t. world. Ground truth, not observed by policy
    self.wTog_real_pos, self.wTog_real_quat = T.mat2pose(np.eye(4))
    n = self.config['num_slots']
    gap = self.config['slot_gap'] + self.config['slot_block_width']
    self.slot_lateral_coords = [i*gap for i in range(-(n//2), n//2+1)]
    self.slot_max_lateral_distance = (n//2 + 0.5) * gap
    # noisy pose of the object coordinate system goal w.r.t. world. kept for convenience
    self.wTog_pos, self.wTog_quat = T.mat2pose(np.eye(4))
    # noisy goal for the end effector pose w.r.t. world
    self.wTeg_pos, self.wTeg_quat = T.mat2pose(np.eye(4))
    # pose of object w.r.t. EEF
    # in practice, we will use a guessed nominal value. Noise will be sampled around it
    self.nominal_oTe_pos = np.array([-0.13, 0.0, 0.0])
    self.nominal_oTe_quat = np.array([-0.5, 0.5, -0.5, 0.5])
    # w.r.t. slot
    self.goal_height = self.slotset.bottom_offset + \
        np.array([0, 0, 1]) * (self.plate.horizontal_radius + self.slot_base_plate_thickness + 0.02)
    # pose of world w.r.t. slot
    self.sTw = np.eye(4)

    # robot initialization data
    if self.config['randomize_eef_location']:
      with open(osp.join('assets', 'init_states.pk'), 'rb') as f:
        self.init_states = pickle.load(f)
    else:
      self.init_states = {
        'joints': [
          np.array([-0.4826866912393867, -0.4055111009152894, 0.374764684069175, -1.947630644134916, 0.1448955433478241, 1.5677628319798846, 0.6505052537713351]), # high
          np.array([-0.4745332622939749, -0.4173033898275949, 0.401064311177558, -2.199562199220353, 0.1634817045162713, 1.8087871844110435, 0.6422974472771739]), # medium
          np.array([-0.4706892114400018, -0.3821354562621636, 0.437327979787199, -2.412691515082357, 0.1794818352200642, 2.0563733944748894, 0.6399711300094529])  # low
         ],
        'wTes': [
          T.pose2mat((np.array([-0.124846148, 0.0, 1.43604274]), np.array([-1, 0, 0, 0]))), # high
          T.pose2mat((np.array([-0.120134709, 0.0, 1.34066785]), np.array([-1, 0, 0, 0]))), # medium
          T.pose2mat((np.array([-0.124825774, 0.0, 1.24594283]), np.array([-1, 0, 0, 0])))  # low
        ]
      }
    self.cached_xpos = np.vstack([wTe[:3, 3] for wTe in self.init_states['wTes']])
    with open(osp.join('assets', 'eTo.json'), 'r') as f:
      self.init_states['eTo'] = np.array(json.load(f)['eTo'])
    with open(osp.join('assets', 'eval_states.json'), 'r') as f:
      if self.config['randomize_eef_location']:
        self.init_states['eval'] = json.load(f)['random_eef_location']
        self.init_states['n_eval'] = len(self.init_states['eval']['pos_offsets'][0])
      else:
        d = json.load(f)['random_goal_location']
        height_offset = table_top - self.slotset.bottom_offset + np.array([0, 0, 1e-3])
        d['goal_pos'] = [np.asarray(dd)+height_offset for dd in d['goal_pos']]
        self.init_states['eval'] = d
        self.init_states['n_eval'] = len(self.init_states['eval']['goal_pos'][0])

    if controller_configs is None:
      controller_configs = load_controller_config(default_controller='OSC_POSE')
      
    self.current_ee_target = np.copy(self.init_states['wTes'][-1][:3, 3])

    super(ContactInsertEnv, self).__init__(
      robots=robots,
      env_configuration=env_configuration,
      controller_configs=controller_configs,
      mount_types="default",
      gripper_types=gripper_types,
      initialization_noise=initialization_noise,
      use_camera_obs=use_camera_obs,
      has_renderer=has_renderer,
      has_offscreen_renderer=has_offscreen_renderer,
      render_camera=render_camera,
      render_collision_mesh=render_collision_mesh,
      render_visual_mesh=render_visual_mesh,
      render_gpu_device_id=render_gpu_device_id,
      control_freq=control_freq,
      horizon=horizon,
      ignore_done=ignore_done,
      hard_reset=hard_reset,
      camera_names=camera_names,
      camera_heights=camera_heights,
      camera_widths=camera_widths,
      camera_depths=camera_depths,
      init_paused=init_paused)
    
    # uncomment the following to visualize the object and goal frames
    # self.viewer.viewer.vopt.frame = 3  # vis sites only
    # self.viewer.viewer.vopt.sitegroup[:] = 0  # no site group, except...
    # self.viewer.viewer.vopt.sitegroup[4] = 1  # site group 2

    self.set_default_action(self.noop_action)
    self.previous_action = self.noop_action
    self._video_path = None
    self._frames_buffer = []

    # some keyboard callback functions for convenient debugging
    if self.has_renderer:
      def print_eef_pose_cb(window, key, scancode, action, mods, env):
        env.print_eef_pose()
      self.viewer.add_keypress_callback(
          glfw.KEY_P, partial(print_eef_pose_cb, env=self))
      def print_robot_joints_cb(window, key, scancode, action, mods, env):
        env.print_robot_joints()
      self.viewer.add_keypress_callback(
          glfw.KEY_J, partial(print_robot_joints_cb, env=self))
      def print_plate_pose_cb(window, key, scancode, action, mods, env):
        env.print_plate_pose()
      self.viewer.add_keypress_callback(
        glfw.KEY_P, partial(print_plate_pose_cb, env=self)
      )
      def print_ee_ft_cb(window, key, scancode, action, mods, env):
        env.print_ee_forcetorque()
      self.viewer.add_keypress_callback(
        glfw.KEY_K, partial(print_ee_ft_cb, env=self)
      )
      def print_and_save_sim_state(window, key, scancode, action, mods, env):
        env.print_and_save_sim_state()
      self.viewer.add_keypress_callback(
        glfw.KEY_PERIOD, partial(print_and_save_sim_state, env=self)
      )
    self.teleop_mode(teleop)

  
  def _construct_slot(self, **object_kwargs):
    c = self.config
    n = c['num_slots']
    geom_locations = []
    geom_sizes = []
    geom_names = []

    # base plate
    base_plate_hsize = [
      self.half_slot_breadth,
      # extra width on either side to avoid toppling
      0.5 * (n*c['slot_gap'] + (n+1)*c['slot_block_width'] + 2*self.slot_base_plate_stability_extension),
      0.5 * self.slot_base_plate_thickness
    ]
    geom_locations.append([0, 0, 0])
    geom_sizes.append(base_plate_hsize)
    geom_names.append('base_plate')

    # vertical blocks
    block_hsize = [self.half_slot_breadth, self.config['slot_block_width']/2.0, self.config['slot_height']/2.0]
    geom_locations += [[
      0,
      self.slot_base_plate_stability_extension + (c['slot_block_width']+c['slot_gap'])*i,
      self.slot_base_plate_thickness
    ] for i in range(n+1)]
    geom_sizes += [block_hsize] * (n+1)
    geom_names += [f'block{i}' for i in range(n+1)]

    total_size = [base_plate_hsize[0], base_plate_hsize[1], base_plate_hsize[2]+block_hsize[2]]
    kwargs = {k: v for k,v in object_kwargs.items() if k != 'friction'}
    slotset = CompositeObject(name='slotset', total_size=total_size, geom_types=['box']*(n+2),
                              geom_locations=geom_locations, geom_sizes=geom_sizes,
                              geom_frictions=[object_kwargs['friction']]*(n+2), rgba=(1,0,0,1), **kwargs)
    # uncomment the following to visualize the goal frame
    # o = slotset.bottom_offset + np.array([0, 0, 1]) \
    #     * (self.plate.horizontal_radius + self.slot_base_plate_thickness + 0.02)
    # o = ' '.join([f'{x}' for x in o])
    # slotset = CompositeObject(name='slotset', total_size=total_size, geom_types=['box']*(n+2),
    #                           geom_locations=geom_locations, geom_sizes=geom_sizes,
    #                           geom_frictions=[object_kwargs['friction']]*(n+2), rgba=(1,0,0,1),
    #                           sites=[{"pos": o, "group": "4"}, ], **kwargs)
    return slotset
  

  def _load_model(self):
    super(ContactInsertEnv, self)._load_model()

    xpos = self.robots[0].robot_model.base_xpos_offset['table'](self.table_full_size[0])
    self.robots[0].robot_model.set_base_xpos(xpos)

    mujoco_arena = TableArena(table_full_size=self.table_full_size,
                              table_friction=self.table_friction,
                              table_offset=self.table_offset)
    mujoco_arena.set_origin([0, 0, 0])

    self.model = ManipulationTask(mujoco_arena=mujoco_arena, mujoco_robots=[
                                  robot.robot_model for robot in self.robots],
                                  mujoco_objects=[self.plate, self.slotset]+self.supports)

  
  ########################################
  # setup functions
  def _setup_references(self):
    super(ContactInsertEnv, self)._setup_references()
    self.table_body_id = self.sim.model.body_name2id('table')
    self.plate_body_id = self.sim.model.body_name2id(self.plate.root_body)
    self.slot_body_id = self.sim.model.body_name2id(self.slotset.root_body)
    self.eef_site_id = self.sim.model.site_name2id(f'{self.robots[0].gripper.naming_prefix}stiffness_frame')
    # plate joint references
    self._ref_object_pos_indexes = []
    self._ref_object_vel_indexes = []
    for joint_name in self.plate.joints:
      idxs = self.sim.model.get_joint_qpos_addr(joint_name)
      self._ref_object_pos_indexes.extend(list(range(*idxs)))
      idxs = self.sim.model.get_joint_qvel_addr(joint_name)
      self._ref_object_vel_indexes.extend(list(range(*idxs)))

  
  def _setup_observables(self):
    observables = super(ContactInsertEnv, self)._setup_observables()
    self.pf = self.robots[0].robot_model.naming_prefix

    @sensor(modality=f'{self.pf}proprio')
    def external_wrench(obs_cache):
      """
      Wrench at site_name generated by contacts in the environment
      We subtract torques generated by frictionloss etc from the total torques, because efc_force[contact_idxs]
      has a complex contact formulation not directly interpretable
      """
      m = self.sim.model
      d = self.sim.data
      efc_types = d.efc_type[:d.nefc]
      idx = np.logical_or(efc_types == 0, np.logical_or(efc_types == 1, efc_types == 3))
      idx = np.arange(d.nefc)[idx]
      xfrc_noncontact = d.efc_force[idx, np.newaxis]
      Jt = d.efc_J[idx, :].T
      qfrc_noncontact = Jt @ xfrc_noncontact
      # negative sign to convert from "applied on robot" to "exerted by robot" torque
      qfrc_contact = qfrc_noncontact - d.qfrc_constraint[:, np.newaxis]
      
      jacp = d.get_site_jacp('gripper0_stiffness_frame').reshape((3, m.nv))
      jacr = d.get_site_jacr('gripper0_stiffness_frame').reshape((3, m.nv))
      J = np.vstack((jacp, jacr))
      F = np.linalg.pinv(J.T) @ qfrc_contact
      F = F.squeeze()
      if sum(abs(F)) < 1e-5:  # wrench is all 0s because grasp is temporarily lost
        F[:] = self.past_external_wrench + self.rng.uniform(-1e-2, 1e-2, size=F.shape)
      self.past_external_wrench[:] = F
      return F

    # next four sensors for pose of EE w.r.t. goal
    @sensor(modality=f'{self.pf}goal')
    def goal_pos(obs_cache):
      return self.wTeg_pos - self._eef_xpos

    #     w2Re = w2Rw1 * w1Re
    # => w2Rw1 = w2Re * eRw1
    # => w2Rw1 = wRg * wRe.T 
    # OSC_POSE rotation action needs w2Rw1 as action
    @sensor(modality=f'{self.pf}goal')
    def goal_rot(obs_cache):
      q = T.quat_multiply(self.wTeg_quat, T.quat_inverse(self._eef_xquat))
      q *= math.copysign(1, q[3])  # quaternion flipping = discontinuity in axis angle representation
      return T.quat2axisangle(q)

    @sensor(modality=f'{self.pf}goal') # used for goaldist_penalty
    def real_object_goal_pos(obs_cache):
      return self.wTog_real_pos - np.array(self.sim.data.body_xpos[self.plate_body_id])

    @sensor(modality=f'{self.pf}goal') # used for goaldist_penalty
    def real_object_goal_rot(obs_cache):
      q_object = T.convert_quat(self.sim.data.body_xquat[self.plate_body_id], to="xyzw")
      q = T.quat_multiply(self.wTog_real_quat, T.quat_inverse(q_object))
      q *= math.copysign(1, q[3])  # quaternion flipping = discontinuity in axis angle representation
      return T.quat2axisangle(q)

    sensors = [external_wrench, goal_pos, goal_rot, real_object_goal_pos, real_object_goal_rot]
    names = [f'{self.pf}external_wrench', 'goal_pos', 'goal_rot', 'real_object_goal_pos', 'real_object_goal_rot']
    self.obs_keys = self.obs_keys or names[:-2]

    if self.use_object_obs:
      @sensor(modality='object')
      def plate_pos(obs_cache):
        return np.array(self.sim.data.body_xpos[self.plate_body_id])

      @sensor(modality='object')
      def plate_quat(obs_cache):
        return T.convert_quat(self.sim.data.body_xquat[self.plate_body_id], to="xyzw")

      sensors += [plate_pos, plate_quat]
      names += ['object_pos', 'object_quat']

    for name, s in zip(names, sensors):
      observables[name] = Observable(
        name=name,
        sensor=s,
        sampling_rate=self.control_freq
      )

    return observables

  
  ########################################
  # reset functions
  def _reset_internal(self):
    # find the closest EEF position from the cache and use its joint angles
    dists = np.linalg.norm(self.cached_xpos - self.current_ee_target[np.newaxis, :], axis=1)
    idx = np.argmin(dists)
    # init_qpos needs to be set because it is used to overwrite sim.data inside _reset_internal()
    self.robots[0].init_qpos[:] = self.init_states['joints'][idx]

    # place plate support(s)
    for sup, pos in zip(self.supports, self.supports_pos):
      self._set_joint_qpos(sup.joints[0], np.hstack((pos, [1, 0, 0, 0])))

    # place slot (overwriting slot pose from the initial state)
    if self.eval_mode:
      idx = self.state_idx % self.n_eval_states if self.sequential_eval else self.rng.integers(self.n_eval_states)
      self.slot_pos = self.init_states['eval']['goal_pos'][self.init_idx][idx]
      slot_rot_deg = self.init_states['eval']['goal_rot_deg'][self.init_idx][idx]
      self.state_idx += 1
    else:
      self.slot_pos = self._sample_uniform(*self.slot_pos_range[self.init_idx]) 
      slot_rot_deg = self._sample_uniform(*self.slot_rot_range_deg[self.init_idx])
    self.slot_rot_quat = T.simple_axangle2quat(np.array([0, 0, 1]), slot_rot_deg)
    self._set_joint_qpos(self.slotset.joints[0],
                         np.hstack((self.slot_pos, T.convert_quat(self.slot_rot_quat, to='wxyz'))))
    self.sTw = np.linalg.inv(T.make_pose(self.slot_pos, T.quat2mat(self.slot_rot_quat)))

    # place plate
    # q = T.convert_quat(T.mat2quat(np.eye(3)), to='wxyz')
    q = np.array([1., 0., 0., 0.], dtype=np.float32)
    self._set_joint_qpos(self.plate.joints[0], np.hstack((self.plate_pos, q)))
    super(ContactInsertEnv, self)._reset_internal()  # calls sim.forward()

    self.episode_started = False
    self.episode_done = False
    self.done_reason = -1

  def _set_joint_qpos(self, joint, qpos):
    self.sim.data.set_joint_qpos(joint, qpos)
    self.sim.data.set_joint_qvel(joint, np.zeros(6))

  def reset(self):
    old_policy_lag_range = self.config['policy_lag_range'][:]
    self.config['policy_lag_range'] = [0, 1]
    n_tries = 0
    self.init_idx = self.rng.choice(3) if self.eval_mode else self.rng.choice(3, p=self.init_idx_p)
    while n_tries < self.config['acquire_grasp_tries']:
      if self.forced_init_idx is not None:
        self.init_idx = self.forced_init_idx
      wTe_target = np.copy(self.init_states['wTes'][self.init_idx])
      wTe_offset = np.eye(4)
      if self.config['randomize_eef_location']:
        if self.eval_mode:
          idx = self.state_idx % self.n_eval_states if self.sequential_eval else self.rng.integers(self.n_eval_states)
          wTe_offset[:3, 3] = self.init_states['eval']['pos_offsets'][self.init_idx][idx]
          wTe_rot_offset = self.init_states['eval']['rot_offsets'][self.init_idx][idx]
          wTe_offset[:3, :3] = T.euler2mat(np.deg2rad(wTe_rot_offset))
        else:
          wTe_offset = np.eye(4)
          wTe_offset[:3, 3] = self._sample_uniform(*self.config['eef_pos_offset_range'][self.init_idx])
          wTe_rot_offset = self._sample_uniform(*self.config['eef_rot_deg_offset_range'][self.init_idx])
          wTe_offset[:3, :3] = T.euler2mat(np.deg2rad(wTe_rot_offset))
        
      wTe_target = wTe_offset @ wTe_target
      self.current_ee_target = np.copy(wTe_target[:3, 3])
      super(ContactInsertEnv, self).reset()

      # open gripper
      self.grip('open', N=2)

      if not self.override_plate_pose:
        # check if plate can be there without colliding
        wTo_target_pos, wTo_target_quat = T.mat2pose(wTe_target @ self.init_states['eTo'])
        self._set_joint_qpos(self.plate.joints[0],
                             np.hstack((wTo_target_pos, T.convert_quat(wTo_target_quat, to="wxyz"))))
        try:
          self.sim.forward()
        except mujoco_py.MujocoException as e:
          self.logger.warn(e)
          n_tries += 1
          continue
        contacts = self.get_contacts(self.plate)
        contacts = [c for c in contacts if not (('robot' in c) or ('gripper' in c))]
        if len(contacts) > 0:
          self.logger.debug(f'sending EEF to\n{np.array2string(wTe_target)}\nwould cause the plate to collide')
          n_tries += 1
          continue

        # place plate back out of the way
        q = T.convert_quat(T.mat2quat(np.eye(3)), to='wxyz')
        self._set_joint_qpos(self.plate.joints[0], np.hstack((self.plate_pos, q)))
        try:
          self.sim.forward()
        except mujoco_py.MujocoException as e:
          self.logger.warn(e)
          n_tries += 1
          continue
      
      # move arm to target location with gripper open
      self.grasp_action = -1  # open
      if not self.move_to_pose_goal(wTe_target[:3, 3], T.mat2quat(wTe_target[:3, :3])):
        self.logger.warning('Could not move EEF to random location')
        n_tries += 1
        continue

      if self.has_renderer:
        self.render()

      # acquire_grasp
      if self._acquire_grasp():
        self.episode_started = True
        self.timestep = 0
        self._store_static_transformations()
        break
      else:
        self.logger.warning("Plate grasp not acquired")
        n_tries += 1
    else:
      self.logger.warning("Max re-tries for grasping plate exceeded, resetting with init_idx = 0")
      self.forced_init_idx = 0
      obs = self.reset()
      self.forced_init_idx = None
      return obs

    if not self._video_path is None:
        self._frames_buffer.append(self._get_frame())
    self.config['policy_lag_range'] = old_policy_lag_range
    self.previous_action = self.noop_action
    return self._get_observations(force_update=True)


  ########################################
  # rewards
  def _check_done(self):
    """
    sets self.episode_done to True if
    1) episode has started properly, and 
    2) object is not in grasp, or
    3) plate (X, Y) is between the two slots, and
    4) it is almost vertical
    Returns object_in_grasp (bool)
    """
    obs = self._get_observations()
    if self._check_object_in_grasp():
      self.grasp_lost_counter = 0
    else:
      self.grasp_lost_counter += 1
    object_in_grasp = self.no_grasp_acquire or (self.grasp_lost_counter < self.config['drop_steps'])
    # self.logger.info(f'pos err = {pos_err:.4f}, rot_err = {np.rad2deg(rot_err):.4f}, '
    #                  'episode started = {self.episode_started}, object in grasp = {object_in_grasp}')
    if self._check_goal_reached(obs):
      self.goal_reached_counter += 1
    else:
      self.goal_reached_counter = 0
    goal_reached = self.goal_reached_counter >= self.goalreach_steps
    self.episode_done = False
    self.done_reason = -1
    if self.episode_started:
      if not object_in_grasp:
        self.episode_done = True
        self.done_reason = 0
      elif goal_reached:
        self.episode_done = True
        self.done_reason = 2
    # done = done and ('table_collision' in self.get_contacts(self.plate))
    if self.episode_done and (not self.ignore_done):
      status = 'Reached goal' if object_in_grasp else 'Lost grasp'
      self.logger.debug(f'{status} at time step = {self.timestep}')
    return object_in_grasp


  def reward(self, action):
    """
    reward is penalty-based rather than incentive-based because the residual policy already incentivizes
    motion towards the goal
    """
    
    # drop penalty
    object_in_grasp = self._check_done()
    drop_p = 0 if object_in_grasp else self.config['drop_penalty']
    
    # goal reaching reward
    if self.done_reason == 2:
      discount = np.power(0.5, np.abs(self.closest_slot)) # reduce reward by a factor of 0.5 each slot away from central slot
      greach_r = self.config['goalreach_reward'] * discount
    else:
      greach_r = 0
    
    obs = self._get_observations()
    
    # goal distance penalty (pos)
    gpos_p = 0
    if self.config['use_goaldist_penalty'] and (self.goal_reached_counter == 0):
      pos_goal_dist = min(np.linalg.norm(obs['real_object_goal_pos']), self.config['goaldist_pos_penalty_cutoff'])
      gpos_p = self.goaldist_pos_penalty_slope * pos_goal_dist
    
    # goal distance penalty (rot)
    grot_p = 0
    if self.config['use_goaldist_penalty'] and (self.goal_reached_counter == 0):
      rot_goal_dist = min(np.linalg.norm(obs['real_object_goal_rot']), self.config['goaldist_rot_penalty_cutoff'])
      grot_p = self.goaldist_rot_penalty_slope * rot_goal_dist
    
    # force penalty
    f_mag = np.linalg.norm(obs[f'{self.pf}external_wrench'][:3])
    f_p = self.config['use_ft_penalty'] * self.config['force_penalty_mag'] / \
        (1.0 + np.exp(self.config['force_penalty_slope'] * (self.config['force_penalty_cutoff']-f_mag)))
    
    # torque penalty
    t_mag = np.linalg.norm(obs[f'{self.pf}external_wrench'][3:])
    t_p = self.config['use_ft_penalty'] * self.config['torque_penalty_mag'] / \
        (1.0 + np.exp(self.config['torque_penalty_slope'] * (self.config['torque_penalty_cutoff']-t_mag)))

    d_action = action[:-1] - self.previous_action[:-1]
    
    # action delta penalty (pos)
    dact_p = 0
    if self.config['use_action_delta_penalty']:
      pos_d_action = min(np.linalg.norm(d_action[:3]), self.config['goaldist_pos_penalty_cutoff'])
      dact_p = self.goaldist_pos_penalty_slope * pos_d_action

    # action delta penalty (rot)
    dact_r = 0
    if self.config['use_action_delta_penalty']:
      rot_d_action = min(np.linalg.norm(d_action[3:]), self.config['goaldist_rot_penalty_cutoff'])
      dact_r = self.goaldist_rot_penalty_slope * rot_d_action
    
    R = greach_r - (self.config['time_penalty'] + drop_p + gpos_p + grot_p - dact_p - dact_r + f_p + t_p)
    
    # if self.episode_started:
    #   self.logger.debug(f'R = {R:.4f}, time = {self.config["time_penalty"]:.4f}, drop = {drop_p:.4f}, goal_pos = {gpos_p:.4f}, goal_rot = {grot_p:.4f}, f = {f_p:.4f}, t = {t_p:.4f}')
        
    return R


  ########################################
  # step functions
  def step(self, input_action, record_frame=True):
    """
    steps environment with the action predicted by the learnt policy
    action = [translation vector(3) rotation axangle(3)]; all elements in [0, 1]
    """
    policy_lag = self.rng.integers(*self.config['policy_lag_range'])
    self.set_policy_lag(policy_lag)

    action = np.copy(input_action)

    w = self.config['residual_weight']
    
    # add to base pos and rot actions
    pos_action = w * action[0:3] * self.config['max_pos_action']
    pos_action += self.base_pos_action
    rot_action = w * action[3:6] * self.config['max_rot_action']
    # # TODO optimize the following 3 lines
    rot_action_quat = T.axisangle2quat(rot_action)
    rot_action_quat = T.quat_multiply(T.axisangle2quat(self.base_rot_action), rot_action_quat)
    rot_action_quat *= math.copysign(1, rot_action_quat[3])  # prevent discontinuity in axis angle representation
    rot_action = T.quat2axisangle(rot_action_quat)


    action[:6] = np.hstack((pos_action, rot_action))
    if self.episode_started and (not self._teleop_mode):
      action[6] = self.grasp_action

    # apply full action
    try:
      obs, reward, time_over, info = super(ContactInsertEnv, self).step(action)
    except mujoco_py.MujocoException as e:
      self.logger.warn(e)
      self.logger.warn(f'Encountered the above exception in step({np.array2string(action)})')
      # crashing MuJoCo is equivalent to dropping the plate
      self.done = self.episode_done = True
      self.done_reason = 0
      return self._get_observations(), -(self.config['time_penalty']+self.config['drop_penalty']), self.done, \
          {'done_reason': self.done_reason}
    
    if self.has_renderer:
      self.render()

    if (not self._video_path is None) and record_frame:
        self._frames_buffer.append(self._get_frame())
    
    self.done = (self.episode_done or time_over) and (not self.ignore_done)
    if self.done:
      if time_over:
        status = 'time over'
        self.done_reason = 1
      else:
        status = 'grasp lost or goal succeeded'
      self.logger.debug(f"Done at timestep {self.timestep} because {status}")

    # store base actions for the next step
    self._calc_base_actions(obs)
    self.previous_action[:] = action

    return obs, reward, self.done, dict(info, done_reason=self.done_reason)


  def _calc_base_actions(self, obs):
    w = self.config['base_weight']
    self.base_pos_action = T.clip_translation(obs['goal_pos'], w*self.config['max_pos_action'])[0]
    self.base_rot_action = T.clip_axangle_rotation(obs['goal_rot'], w*self.config['max_rot_action_deg'])[0]


  ########################################
  # pre defined motions
  def _acquire_grasp(self):
    """
    Holds object (plate) in current pose, and closes the gripper to acquire grasp
    """
    if self.no_grasp_acquire:
      return True
    
    wTe = np.eye(4)
    wTe[:3, 3] = self._eef_xpos
    wTe[:3, :3] = self._eef_xmat
    wTo_pos, wTo_quat = T.mat2pose(wTe @ self.init_states['eTo'])
    p = np.hstack((wTo_pos, T.convert_quat(wTo_quat, to="wxyz")))
    done = self.grip('close', plate_pose=p)
    return done and self._check_object_in_grasp()


  def follow_waypoints(self, wTegs):
    done = True
    for i, wTeg in enumerate(wTegs):
      self.logger.debug('Waypoint {:d}/{:d}'.format(i+1, len(wTegs)))
      if isinstance(wTeg, tuple):
        wTeg_pos, wTeg_quat = wTeg
        done = done and self.move_to_pose_goal(wTeg_pos, wTeg_quat)
      elif wTeg in ('close', 'open'):
        done = done and self.grip(wTeg)
      else:
        raise NotImplementedError
      contacting_geoms = ' '.join(self.get_contacts(self.plate))
      self.logger.debug(f'Plate contacting: {contacting_geoms}')
      if not done:
        break
    else:
      self.logger.debug('Waypoints done')
    return done
      

  def move_to_pose_goal(self, wTeg_pos, wTeg_quat, N=500):
    # backup some state
    prev_r, prev_b = self.config['residual_weight'], self.config['base_weight']
    self.config['residual_weight'] = 0.0
    self.config['base_weight'] = 1.0
    self.ignore_done = True
    prev_wTeg = (np.copy(self.wTeg_pos), np.copy(self.wTeg_quat))
    self.set_goal(wTeg_pos, wTeg_quat)

    done = False    
    for _ in range(N):
      try:
        obs, _, _, _ = self.step(self.noop_action)
      except mujoco_py.MujocoException as e:
        self.logger.warn(e)
        self.logger.warn('Encountered the above exception in move_to_pose_goal()')
        self.logger.warn(f'base_pos_action = {np.array2string(self.base_pos_action)}')
        self.logger.warn(f'base_rot_action = {np.array2string(self.base_rot_action)}')
        break

      pos_err_mag = np.linalg.norm(obs['goal_pos'])
      rot_err_mag = np.rad2deg(np.linalg.norm(obs['goal_rot']))
      rot_err_mag = min(rot_err_mag, 360.0-rot_err_mag)
      if (pos_err_mag < self.config['pos_tolerance']) and (rot_err_mag < self.config['rot_tolerance_deg']):
        self.logger.debug('reached waypoint')
        done = True
        break
    
    # restore backup
    self.config['residual_weight'] = prev_r
    self.config['base_weight'] = prev_b
    self.ignore_done = False
    self.set_goal(*prev_wTeg)
    return done


  def grip(self, state, tol=0, plate_pose=None, N=None):
    if state == 'close':
      self.grasp_action = 1
      N = N or 10
    elif state == 'open':
      self.grasp_action = -1
      N = N or 10
    else:
      raise NameError
    self.ignore_done = True
    self.logger.debug(f'{state} gripper')

    # induce no EEF motion
    prev_wTeg = (np.copy(self.wTeg_pos), np.copy(self.wTeg_quat))
    self.set_goal(self._eef_xpos, self._eef_xquat)
    
    prev_obs = None
    done = True
    for i in range(N):
      if plate_pose is not None:
        self._set_joint_qpos(self.plate.joints[0], plate_pose)
        self.sim.data.qfrc_applied[self._ref_object_vel_indexes] = self.sim.data.qfrc_bias[self._ref_object_vel_indexes]
      try:
        self.step(self.noop_action, record_frame=False)
      except mujoco_py.MujocoException as e:
        self.logger.warn(e)
        self.logger.warn(f'Encountered the above exception in grip({state})')
        self.logger.warn(f'base_pos_action = {np.array2string(self.base_pos_action)}')
        self.logger.warn(f'base_rot_action = {np.array2string(self.base_rot_action)}')
        done = False
        break
      obs = self._observables[f'{self.pf}gripper_qpos'].obs
      if (i>0) and (np.linalg.norm(obs-prev_obs)<tol):
        done = True
        break
      prev_obs = np.copy(obs)
    self.sim.data.qfrc_applied[self._ref_object_vel_indexes] = 0.

    # restore previous goals
    self.set_goal(*prev_wTeg)
    self.ignore_done = False
    return done

  
  def set_goal(self, wTeg_pos, wTeg_quat):
    """
    Overrides the goal for residual base EEF motion
    wTg: 4x4 pose of the goal w.r.t. world
    """
    self.wTeg_pos, self.wTeg_quat = wTeg_pos, wTeg_quat
    # update the observables, since the goal has changed
    for observable in self._observables.values():
      observable.reset()
      observable._sampled = False
    self._update_observables()
    obs = self._get_observations()
    self._calc_base_actions(obs)


  ########################################
  # convenience functions
  def _store_static_transformations(self):
    """
    - stores real and noisy transformations between w (world), o (object/plate), e (end effector), og (object goal)
    and eg (end effector goal)
    - oTe_real is assumed to not change during an episode
    - call this funtion after slot is in the correct location, eef is in start location (i.e. RL env T=0), and
    object/plate is grasped by the eef
    - real = noiseless
    """
    
    # sample noise in slot pose
    slot_pos_noise = self._sample_uniform(*self.config['slot_pos_noise_range'], self.noise_level)
    slot_rot_noise_deg = self._sample_uniform(*self.config['slot_rot_noise_range_deg'], self.noise_level)
    slot_rot_noise_quat = T.simple_axangle2quat(np.array([0, 0, 1]), slot_rot_noise_deg)

    # calculate noisy slot pose
    noisy_slot_pos, noisy_slot_rot_quat = self._apply_noise(self.slot_pos, self.slot_rot_quat, slot_pos_noise,
                                                            slot_rot_noise_quat)

    # transform from slot frame to object goal frame
    # q = T.quat_multiply(T.simple_axangle2quat(np.array([0, 1, 0]), 90), T.simple_axangle2quat(np.array([1, 0, 0]), -90))
    q = np.array([-0.5,  0.5,  0.5,  0.5], dtype=np.float32)
    self.wTog_real_pos = self.slot_pos + self.goal_height
    self.wTog_real_quat = T.quat_multiply(self.slot_rot_quat, q)
    self.wTog_pos = noisy_slot_pos + self.goal_height
    self.wTog_quat = T.quat_multiply(noisy_slot_rot_quat, q)

    # oTe_real
    obs = self._get_observations(force_update=True)
    wTo_real_mat = T.pose2mat((obs['object_pos'], obs['object_quat']))
    wTe_real_mat = np.eye(4)
    wTe_real_mat[:3, :3] = self._eef_xmat
    wTe_real_mat[:3, 3]  = self._eef_xpos
    oTe_real_mat = np.linalg.inv(wTo_real_mat) @ wTe_real_mat

    # caching to speed up EEF randomization
    if self.config['randomize_eef_location']:
      init_q_pos = self.sim.data.qpos[self.robots[0].controller.qpos_index]
      self.init_states['joints'].append(init_q_pos)
      self.init_states['wTes'].append(wTe_real_mat)
      self.cached_xpos = np.vstack((self.cached_xpos, wTe_real_mat[:3, 3]))
    
    if self.config['noisy_eef']:
      # sample noise in object pose w.r.t. EEF
      oTe_pos_noise = self._sample_uniform(*self.config['eef_pos_noise_range'], self.noise_level)
      oTe_rot_noise = np.deg2rad(self._sample_uniform(*self.config['eef_rot_noise_range_deg'], self.noise_level))
      oTe_rot_noise_quat = T.mat2quat(T.euler2mat(oTe_rot_noise))
      oTe_mat = self._apply_noise(self.nominal_oTe_pos, self.nominal_oTe_quat, oTe_pos_noise, oTe_rot_noise_quat)
    else:
      oTe_mat = np.copy(oTe_real_mat)
    
    # restore the original goal
    wTeg_mat = T.make_pose(self.wTog_pos, T.quat2mat(self.wTog_quat)) @ oTe_mat
    wTeg_pos, wTeg_quat = T.mat2pose(wTeg_mat)
    self.set_goal(wTeg_pos, wTeg_quat)


  def _check_object_in_grasp(self):
    # check whether grasp was successfully acquired
    # TODO: why are the finger pads not in contact even when the fingers are in contact?
    # return self._check_grasp(self.robots[0].robot_model.eef_name, 'plate')
    return self._check_grasp(
      [self.robots[0].gripper.important_geoms['left_finger'], self.robots[0].gripper.important_geoms['right_finger']],
      self.plate
    )


  def _check_goal_reached(self, obs):
    # transform object pose to the slot coordinates
    w_x = np.ones((4, 1))
    w_x[:3, 0] = obs['object_pos']
    s_x = self.sTw @ w_x
    s_x = s_x[:3, 0]

    # filter by height, medial distance, and max lateral distance
    if (abs(s_x[0]) > (self.half_slot_breadth - 0.1)) or (abs(s_x[1]) > self.slot_max_lateral_distance) \
            or (s_x[2] > self.goal_height[2] + self.config['success_pos_thresh']):
      self.closest_slot = None
      return False

    # find closest slot
    self.closest_slot = np.argmin(np.abs(self.slot_lateral_coords - s_x[1])) - self.config['num_slots']//2
    # self.logger.info(f'reached slot {self.closest_slot} at timestep {self.timestep}')
    return True


  @property
  def noop_action(self):
    a = np.zeros(self.action_dim)
    # a[:3] = self.base_pos_action
    # a[3:6] = self.base_rot_action
    a[-1] = self.grasp_action
    return a


  @property
  def _eef_xpos(self):
    return np.array(self.sim.data.site_xpos[self.eef_site_id])


  @property
  def _eef_xmat(self):
    return np.array(self.sim.data.site_xmat[self.eef_site_id]).reshape(3, 3)


  def get_slot_pose(self):
    return {'pos': copy.deepcopy(self.slot_pos), 'rot_deg': np.rad2deg(T.quat2axisangle(self.slot_rot_quat)[2])}

  
  def get_robot_joint_values(self):
    return copy.deepcopy(self.robots[0]._joint_positions)

  
  def print_eef_pose(self):
    self.logger.info('EEF pos = [{:.5f}, {:.5f}, {:.5f}]'.format(*self._eef_xpos))
    self.logger.info('EEF quat = [{:.5f}, {:.5f}, {:.5f}, {:.5f}]'.format(*self._eef_xquat))

  
  def print_robot_joints(self):
    joint_values = self.get_robot_joint_values()
    s = ', '.join(['{:.5f}']*len(joint_values))
    s = s.format(*joint_values)
    self.logger.info(f'Robot joints = {s}')
    s = '{:.5f} {:.5f}'.format(*[self.sim.data.qpos[x] for x in self.robots[0]._ref_gripper_joint_pos_indexes])
    self.logger.info(f'Gripper joints = {s}')


  def print_plate_pose(self):
    x = np.array(self.sim.data.body_xpos[self.plate_body_id])
    self.logger.info('Plate pos = {:.5f}, {:.5f}, {:.5f}'.format(*x)) 
    x = np.array(self.sim.data.body_xquat[self.plate_body_id])
    self.logger.info('Plate quat = {:.5f}, {:.5f}, {:.5f}, {:.5f}'.format(*x))


  def print_ee_forcetorque(self):
    obs = self._get_observations()
    f, t = np.split(obs[f'{self.pf}external_wrench'], 2)
    self.logger.info('EEF Force =  {:.5f}, {:.5f}, {:.5f}'.format(*f))
    self.logger.info('EEF Torque = {:.5f}, {:.5f}, {:.5f}'.format(*t))

  
  def print_and_save_sim_state(self):
    """
    see http://www.mujoco.org/book/programming.html#siStateControl
    na = nmocap = nuserdata = 0
    """
    state = {}
    for field in ('qpos', 'qvel', 'qacc_warmstart', 'ctrl', 'qfrc_applied', 'xfrc_applied'):
      state[field] = np.copy(getattr(self.sim.data, field))
    for field in ('_ref_gripper_joint_pos_indexes', '_ref_gripper_joint_vel_indexes', '_ref_joint_pos_indexes',
                  '_ref_joint_vel_indexes'):
      state[field] = np.copy(getattr(self.robots[0], field))
    wTe = np.eye(4)
    wTe[:3, 3]  = self._eef_xpos
    wTe[:3, :3] = self._eef_xmat
    state['wTe'] = wTe
    for k,v in state.items():
      self.logger.info(f'{k}: {np.array2string(v)}')
    filename = osp.join('assets', 'init_state_left.pk')
    with open(filename, 'wb') as f:
      pickle.dump(state, f)
    self.logger.info(f'{filename} written')


  def teleop_mode(self, mode):
    self._teleop_mode = mode
    if mode:
      self.config['residual_weight_bak'] = self.config['residual_weight']
      self.config['base_weight_bak'] = self.config['base_weight']
      self.config['residual_weight'] = 1.0
      self.config['base_weight'] = 0.0
      self.base_pos_action = np.zeros(3)
      self.base_rot_action = np.zeros(3)
      self.ignore_done = True
    else:
      self.config['residual_weight'] = self.config['residual_weight_bak']
      self.config['base_weight'] = self.config['base_weight_bak']
      self.ignore_done = False
      self._update_observables()
      obs = self._get_observations()
      self._calc_base_actions(obs)

  
  def _sample_uniform(self, low, high, f=1.0):
    return self.rng.uniform(f*np.asarray(low), f*np.asarray(high))


  @staticmethod
  def _apply_noise(aTb_pos, aTb_quat, bTn_pos, bTn_quat, return_mat=False):
    """
    noisy version of aTb, aTn = aTb * bTn
    """
    aTb = T.make_pose(aTb_pos, T.quat2mat(aTb_quat))
    bTn = T.make_pose(bTn_pos, T.quat2mat(bTn_quat))
    aTn = np.dot(aTb, bTn)
    return aTn if return_mat else T.mat2pose(aTn)


  def set_state(self, state_dict: dict):
    """
    change some attributes of the env. Expects a dict whose keys are attribute names and values are new attribute
    values
    raises KeyError if attribute is not found
    """
    for k,v in state_dict.items():
      if hasattr(self, k):
        setattr(self, k, v)
      elif k in self.config:
        self.config[k] = v
      else:
        raise NotImplementedError(f'DishwasherEnv or its config dict does not have {k}')
      
  
  def get_state(self) -> dict:
    """
    returns a dict containing the environment state. Currently, only the config dictionary.
    """
    return self.config


  @property
  def n_eval_states(self):
    return self.init_states['n_eval']
  
  
  ########################################
  # video recording functions
  def _get_frame(self):
    return self.sim.render(1920, 1080, camera_name='frontview')[::-1,...]

  
  def start_recording(self, video_path):
    """
    Frames are added to buffer at every call of step
    After calling start_recording, must call finish_recording at the end
    """
    # create render context
    self.sim.render(1920, 1080, camera_name='frontview')
    # make collision geometries (geomgroup 0) invisible
    self.sim._render_context_offscreen.vopt.geomgroup[0] = 0
    self._video_path = video_path

  
  def end_recording(self, clear_frames_buffer = True):
    """
    Pass clear_frames_buffer = False if you wish to retain the frames after recording is finished.
    E.g. to combine frames from different episodes
    """
    writer = imageio.get_writer(self._video_path, fps=int(self.control_freq), macro_block_size=None,
                                ffmpeg_params = ['-s','1920x1080'])
    for f in self._frames_buffer:
        writer.append_data(f)
    writer.close()
    self._video_path = None
    if clear_frames_buffer:
      self._frames_buffer = []


if __name__ == '__main__':
  import init_paths
  import argparse
  import utilities as sutils
  import time
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--no_render', action='store_false', dest='has_renderer')
  parser.add_argument('--video_path', type=str, default=None, help='Path to save video recording. E.g. "video.mp4"')
  parser.add_argument('--config', '-c', required=True, help="config file")
  args = parser.parse_args()
  
  with open(args.config, 'r') as f:
    config = json.load(f)
  
  # # zero noise
  config['slot_pos_noise_range'][0][0] = 0
  config['slot_pos_noise_range'][0][1] = 0
  config['slot_pos_noise_range'][1][0] = 0
  config['slot_pos_noise_range'][1][1] = 0
  config['slot_rot_noise_range_deg'][0] = 0
  config['slot_rot_noise_range_deg'][1] = 0

  sutils.setup_logging(filename=osp.join('logs', 'dishwasher.txt'), level=logging.DEBUG)

  env = ContactInsertEnv(has_renderer=args.has_renderer, init_paused=False, **config, eval_mode=False)
  
  # step through each frame, also set init_paused above to True
  # env.reset()
  # env.teleop_mode(True)
  # while True:
  #   env.step(env.noop_action)

  # time the env stepping frequency
  # print('timing env.step()...')
  # from timeit import default_timer as timer
  # env.reset()
  # count = 0
  # t = 0.0
  # N = 100
  # while count < N:
  #   start = timer()
  #   _, _, done, _ = env.step(env.noop_action)
  #   end = timer()
  #   t += (end - start)
  #   count += 1
  #   if done:
  #     print('resetting...')
  #     env.reset()
  #     print('done')
  # print(f'Stepping frequency = {N/t:.4f} Hz')

  # check multiple resets
  N = 4
  steps = 0
  reset_time = 0
  rollout_time = 0

  if not args.video_path is None:
    env.start_recording(args.video_path)

  for r in range(N):
    print(f'\nReset {r+1}/{N}\n')
    t0 = time.time()
    env.reset()
    t1 = time.time()
    # env.teleop_mode(True)
    done = False
    episode_return = 0

    while not done:
      obs, reward, done, info = env.step(env.noop_action)
      steps += 1
      episode_return += reward
      if done:
        print(info)
    t2 = time.time()
    reset_time += (t1-t0)
    rollout_time += (t2-t1)
    print(f'Episode return {episode_return}')
  total_time = reset_time + rollout_time
  reset_frac = np.round(reset_time/total_time, 5)
  rollout_frac = np.round(rollout_time/total_time, 5)
  fps = steps/total_time
  print(f'FPS = {np.round(fps, 2)}, Reset frac = {np.round(reset_time/total_time, 2)}, '
        f'Rollout frac = {np.round(rollout_time/total_time, 2)}')
  print(f'Steps = {steps}')

  if not args.video_path is None:
    env.end_recording()