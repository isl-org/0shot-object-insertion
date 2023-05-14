from abc import ABC, abstractmethod
import logging
import os
import reverb
import tensorflow as tf

from tf_agents.agents.sac.sac_agent import SacAgent, std_clip_transform
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.metrics.py_metrics import EnvironmentSteps
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks.normal_projection_network import NormalProjectionNetwork
from tf_agents.replay_buffers.reverb_replay_buffer import DEFAULT_TABLE
from tf_agents.replay_buffers.reverb_utils import ReverbAddTrajectoryObserver
from tf_agents.train.actor import collect_metrics, eval_metrics

from tfagents_system.utils.metric import StepsWrittenMetric
from tfagents_system.utils.critic_network import CriticNetwork
from tfagents_system.utils.critic_rnn_network import CriticRnnNetwork


class TrainParamsBase(ABC):
  def __init__(self, env_config: dict):
    self.env_config = env_config
    self.compute_config = dict(n_parallel_workers=int(os.environ.get('SAC_NUM_COLLECT_WORKERS', 1)))
    self.seed_sac_buffer = True
    self.recurrent = False
    
    self.discount = 0.99

    self.critic_learning_rate = 3e-4
    self.actor_learning_rate = 3e-4
    self.alpha_learning_rate = 3e-4
    self.target_update_tau = 0.005
    self.target_update_period = 1
    self.gradient_clipping = None
    self.sac_log_alpha = 1.0

    self.actor_fc_layer_params = (256, 256)
    self.critic_joint_fc_layer_params = (256, 256)

    self.num_iter = 1000
    self.num_steps_per_iter = 50  # number of SGD steps
    self.n_episodes_in_buffer = 350

    self.n_train_episodes_per_iter = 1
    self.n_eval_episodes = 5
    self.n_buffer_init_episodes = 30
    self.eval_period_itrs = 250
    self.prefetch_batches = 3
    self.recurrent_sequence_length = 20

    # number of samples of size train_subepisode_length to sample from replay buffer for 1 round of training
    self.train_batch_size = 256
    self.log_period_itrs = 10  
    self.checkpoint_period_itrs = 250

    # preprocessing combiner for actor and critic
    self.preprocessing_combiner = None

    self.logger = logging.getLogger(__name__)


  @abstractmethod
  def env_ctor(self, seed=None, *args, **kwargs) -> PyEnvironment:
    """
    build and return a PyEnvironment. Use the `self.env_config` dictionary for configuring the environment.
    For more details see https://www.tensorflow.org/agents/tutorials/2_environments_tutorial#python_environments
    and https://www.tensorflow.org/agents/api_docs/python/tf_agents/environments/PyEnvironment.
    """


  def collect_env_args_kwargs(self, *args, **kwargs):
    """
    return args (tuple) and kwargs (dict)
    When creating the experience collection environments, they will be used as TrainParamsBase.env_ctor(*args, **kwargs)
    Use `self.env_config`
    """
    return (), {}
  
  
  def eval_env_args_kwargs(self, *args, **kwargs):
    """
    return args (tuple) and kwargs (dict)
    When creating the evaluation environments, they will be used as TrainParamsBase.env_ctor(*args, **kwargs)
    Use `self.env_config`
    """
    return (), {}
  
  
  def actor_ctor(self, observation_spec, action_spec):
    def normal_projection_net(action_spec, init_means_output_factor=0.5):
      return NormalProjectionNetwork(action_spec, mean_transform=None, state_dependent_std=True,
                                     init_means_output_factor=init_means_output_factor,
                                     std_transform=std_clip_transform, scale_distribution=True)
    if self.recurrent:
      actor_net = ActorDistributionRnnNetwork(observation_spec, action_spec,
                                              preprocessing_combiner=self.preprocessing_combiner,
                                              input_fc_layer_params=(self.actor_fc_layer_params[0],),
                                              lstm_size=(self.actor_fc_layer_params[0],),
                                              output_fc_layer_params=(self.actor_fc_layer_params[1],),
                                              continuous_projection_net=normal_projection_net)
    else:
      actor_net = ActorDistributionNetwork(observation_spec, action_spec,
                                           preprocessing_combiner=self.preprocessing_combiner,
                                           fc_layer_params=self.actor_fc_layer_params,
                                           continuous_projection_net=normal_projection_net)
    return actor_net


  def critic_ctor(self, observation_spec, action_spec):
    if self.recurrent:
      critic_net = CriticRnnNetwork((observation_spec, action_spec),
                                    observation_preprocessing_combiner=self.preprocessing_combiner,
                                    observation_fc_layer_params=None,
                                    action_fc_layer_params=None,
                                    joint_fc_layer_params=(self.critic_joint_fc_layer_params[0],),
                                    lstm_size=(self.critic_joint_fc_layer_params[1],),
                                    joint_preprocessing_combiner=tf.keras.layers.Concatenate(axis=-1))
    else:
      critic_net = CriticNetwork((observation_spec, action_spec),
                                 observation_preprocessing_combiner=self.preprocessing_combiner,
                                 observation_fc_layer_params=None, action_fc_layer_params=None,
                                 joint_fc_layer_params=self.critic_joint_fc_layer_params)
    return critic_net

  
  def train_observers_ctor(self, reverb_port, collect_only=False):
    """
    all 3 - observers, metrics, and transition observers are executed, but only metrics are logged and summarized
    """
    reverb_client = reverb.Client(f'localhost:{reverb_port}')
    sequence_length = self.recurrent_sequence_length if self.recurrent else 2
    observers = dict(
      replay_buffer=ReverbAddTrajectoryObserver(reverb_client, table_name=DEFAULT_TABLE,
                                                sequence_length=sequence_length, stride_length=1),
      # need a large buffer size because there can be many small episodes that end quickly
      steps_written=StepsWrittenMetric(min_length_to_write=sequence_length, buffer_size=self.env_config['horizon'])
    )
    if collect_only:
      return observers, {}, {}
    buffer_size = self.log_period_itrs * self.n_train_episodes_per_iter
    metrics = collect_metrics(buffer_size=buffer_size)
    metrics = dict(reward=metrics[2], episode_length=metrics[3], steps=metrics[1], episodes=metrics[0])
    return observers, {}, metrics


  def eval_observers_ctor(self):
    metrics = eval_metrics(buffer_size=self.n_eval_episodes)
    metrics = dict(reward=metrics[0], episode_length=metrics[1], env_steps=EnvironmentSteps())
    return {}, {}, metrics


  def agent_ctor(self, train_step, obs_spec, action_spec, time_step_spec):
    return SacAgent(
        time_step_spec,
        action_spec,
        actor_network=self.actor_ctor(obs_spec, action_spec),
        critic_network=self.critic_ctor(obs_spec, action_spec),
        actor_optimizer=tf.keras.optimizers.Adam(learning_rate=self.actor_learning_rate),
        critic_optimizer=tf.keras.optimizers.Adam(learning_rate=self.critic_learning_rate),
        alpha_optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha_learning_rate),
        target_update_tau=self.target_update_tau,
        target_update_period=self.target_update_period,
        td_errors_loss_fn=tf.math.squared_difference,
        gamma=self.discount,
        reward_scale_factor=1.0,
        gradient_clipping=self.gradient_clipping,
        train_step_counter=train_step,
        initial_log_alpha=self.sac_log_alpha,
    )
