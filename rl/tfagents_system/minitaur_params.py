from tfagents_system.train_params_base import TrainParamsBase
from tf_agents.environments import suite_pybullet, PyEnvironment

class TrainParams(TrainParamsBase):
  def __init__(self, env_config: dict):
    super(TrainParams, self).__init__(env_config)
    self.num_iter = 1000 # 1M env steps
    self.n_episodes_in_buffer = 350 # 10k env steps
    self.n_buffer_init_episodes = 10 # 10k env steps

    self.n_train_episodes_per_iter = 1
    self.num_steps_per_iter = 500  # number of SGD steps

    # number of samples of size train_subepisode_length to sample from replay buffer for 1 round of training
    self.train_batch_size = 256
    self.log_period_itrs = 5
    self.checkpoint_period_itrs = 5
    self.eval_period_itrs = 5

  
  def env_ctor(self, *args, **kwargs):
    seed = kwargs.pop('seed', None)
    env: PyEnvironment = suite_pybullet.load(*args, **kwargs)
    if seed is not None:
      env.seed(int(seed))
    return env

  
  def collect_env_args_kwargs(self):
    return ('MinitaurBulletEnv-v0', ), dict(max_episode_steps=self.env_config['horizon'])

  
  def eval_env_args_kwargs(self):
    return ('MinitaurBulletEnv-v0', ), dict(max_episode_steps=self.env_config['horizon'])