import collections
import numpy as np
from tf_agents.drivers import py_driver
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.metrics.py_metric import PyStepMetric, PyMetric
from tf_agents.metrics.py_metrics import NumpyDeque, AverageEpisodeLengthMetric
from tf_agents.policies.py_policy import PyPolicy
from tf_agents.trajectories import TimeStep
from tf_agents.utils.nest_utils import batch_nested_array
from typing import Iterable


def compute(metrics: Iterable[PyStepMetric],
            transition_metrics: Iterable[PyStepMetric],
            environment: PyEnvironment,
            policy: PyPolicy,
            num_episodes=1):
  """Compute metrics using `policy` on the `environment`.

  Args:
    metrics: List of metrics to compute.
    environment: py_environment instance.
    policy: py_policy instance used to step the environment. A tf_policy can be
      used in_eager_mode.
    num_episodes: Number of episodes to compute the metrics over.

  Returns:
    A dictionary of results {metric_name: metric_value}
  """
  for metric in metrics:
    metric.reset()
  for tmetric in transition_metrics:
    tmetric.reset()
  time_step = environment.reset()
  policy_state = policy.get_initial_state(environment.batch_size or 1)

  driver = py_driver.PyDriver(
      environment,
      policy,
      observers=metrics,
      transition_observers=transition_metrics,
      max_steps=None,
      max_episodes=num_episodes)
  driver.run(time_step, policy_state)

  results = [(metric.name, metric.result()) for metric in metrics]
  results.extend([(metric.name, metric.result()) for metric in transition_metrics])
  return collections.OrderedDict(results)


class SuccessRateMetric(PyMetric):
  def __init__(self, name='SuccessRate', buffer_size: int=10, batch_size=None):
    """
    Tracks success rate. To use this metric, your PyEnvironment's observation should be a dict with the structure:
    observation['obs']: actual observation
    observation['aux']: auxiliary information dict e.g. "info" field of a Gym environment's step() or reset()
    observation['aux']['successful']: single element np array of dtype np.int32, 0=episode ended in failure, 1=success 
    """
    super(SuccessRateMetric, self).__init__(name)
    self._buffer = NumpyDeque(maxlen=buffer_size, dtype=np.float64)
    self._batch_size = batch_size
    self.reset()

  def reset(self):
    self._buffer.clear()

  def result(self) -> np.float32:
    """Returns the value of this metric."""
    if self._buffer:
      return self._buffer.mean(dtype=np.float32)
    return np.array(0.0, dtype=np.float32)

  def call(self, transition: tuple):
    time_step: TimeStep = transition[0]
    if not self._batch_size:
      if time_step.step_type.ndim == 0:
        self._batch_size = 1
      else:
        assert time_step.step_type.ndim == 1
        self._batch_size = time_step.step_type.shape[0]
      self.reset()
    if time_step.step_type.ndim == 0:
      transition = batch_nested_array(transition)
    self._batched_call(transition)

  def _batched_call(self, transition: tuple):
    next_time_step: TimeStep = transition[2]
    success = (next_time_step.observation['aux']['successful'][next_time_step.is_last()] == 1)
    self._buffer.extend(success)


class StepsWrittenMetric(AverageEpisodeLengthMetric):
  def __init__(self, name='StepsWritten', min_length_to_write: int=2, buffer_size: int=10, batch_size=None):
    """
    if writing an episode to the replay buffer requires it to have at least min_length_to_write steps, returns
    the total number of steps written
    """
    self._min_length_to_write = min_length_to_write
    super(StepsWrittenMetric, self).__init__(name, buffer_size, batch_size)

  def result(self) -> np.float32:
    if self.data:
      episode_lengths = self.data._buffer
      idx = episode_lengths >= self._min_length_to_write
      return np.sum(episode_lengths[idx])
    return np.array(0.0, dtype=np.float32)