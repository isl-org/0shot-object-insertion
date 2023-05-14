"""
Copied from agents/tf_agents/agents/ddpg/critic_rnn_network.py
TODO(samarth): remove this file once upstream agents includes EncodingNetworks
"""
import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.keras_layers import dynamic_unroll_layer
from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.networks.encoding_network import EncodingNetwork
from tf_agents.networks.lstm_encoding_network import LSTMEncodingNetwork
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step
from tf_agents.utils import nest_utils

KERAS_LSTM_FUSED_IMPLEMENTATION = 2


@gin.configurable
class CriticRnnNetwork(network.Network):
  """Creates a recurrent Critic network."""

  def __init__(self,
               input_tensor_spec,
               observation_preprocessing_layers=None,
               observation_preprocessing_combiner=None,
               observation_conv_layer_params=None,
               observation_fc_layer_params=(200,),
               observation_dropout_layer_params=None,
               action_preprocessing_layers=None,
               action_preprocessing_combiner=None,
               action_fc_layer_params=(200,),
               action_dropout_layer_params=None,
               joint_preprocessing_layers=None,
               joint_preprocessing_combiner=None,
               joint_fc_layer_params=(100,),
               lstm_size=None,
               output_fc_layer_params=(200, 100),
               activation_fn=tf.keras.activations.relu,
               kernel_initializer=None,
               last_kernel_initializer=None,
               rnn_construction_fn=None,
               rnn_construction_kwargs=None,
               name='CriticRnnNetwork'):
    """Creates an instance of `CriticRnnNetwork`.

    Args:
      input_tensor_spec: A tuple of (observation, action) each of type
        `tensor_spec.TensorSpec` representing the inputs.
      observation_conv_layer_params: Optional list of convolution layers
        parameters to apply to the observations, where each item is a
        length-three tuple indicating (filters, kernel_size, stride).
      observation_fc_layer_params: Optional list of fully_connected parameters,
        where each item is the number of units in the layer. This is applied
        after the observation convultional layer.
      action_fc_layer_params: Optional list of parameters for a fully_connected
        layer to apply to the actions, where each item is the number of units
        in the layer.
      joint_fc_layer_params: Optional list of parameters for a fully_connected
        layer to apply after merging observations and actions, where each item
        is the number of units in the layer.
      lstm_size: An iterable of ints specifying the LSTM cell sizes to use.
      output_fc_layer_params: Optional list of fully_connected parameters, where
        each item is the number of units in the layer. This is applied after the
        LSTM cell.
      activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
      kernel_initializer: kernel initializer for all layers except for the value
        regression layer. If None, a VarianceScaling initializer will be used.
      last_kernel_initializer: kernel initializer for the value regression layer
        . If None, a RandomUniform initializer will be used.
      rnn_construction_fn: (Optional.) Alternate RNN construction function, e.g.
        tf.keras.layers.LSTM, tf.keras.layers.CuDNNLSTM. It is invalid to
        provide both rnn_construction_fn and lstm_size.
      rnn_construction_kwargs: (Optional.) Dictionary or arguments to pass to
        rnn_construction_fn.

        The RNN will be constructed via:

        ```
        rnn_layer = rnn_construction_fn(**rnn_construction_kwargs)
        ```
      name: A string representing name of the network.

    Raises:
      ValueError: If `observation_spec` or `action_spec` contains more than one
        item.
      ValueError: If neither `lstm_size` nor `rnn_construction_fn` are provided.
      ValueError: If both `lstm_size` and `rnn_construction_fn` are provided.
    """
    if lstm_size is None and rnn_construction_fn is None:
      raise ValueError('Need to provide either custom rnn_construction_fn or '
                       'lstm_size.')
    if lstm_size and rnn_construction_fn:
      raise ValueError('Cannot provide both custom rnn_construction_fn and '
                       'lstm_size.')

    observation_spec, action_spec = input_tensor_spec

    if kernel_initializer is None:
      kernel_initializer = tf.compat.v1.keras.initializers.VarianceScaling(
          scale=1. / 3., mode='fan_in', distribution='uniform')
    if last_kernel_initializer is None:
      last_kernel_initializer = tf.keras.initializers.RandomUniform(
          minval=-0.003, maxval=0.003)

    observation_layers = EncodingNetwork(
        observation_spec,
        observation_preprocessing_layers,
        observation_preprocessing_combiner,
        observation_conv_layer_params,
        observation_fc_layer_params,
        observation_dropout_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=kernel_initializer,
        name='observation_encoding')
    observation_spec = observation_layers.create_variables() 

    action_layers = EncodingNetwork(
        action_spec,
        action_preprocessing_layers,
        action_preprocessing_combiner,
        None,
        action_fc_layer_params,
        action_dropout_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=kernel_initializer,
        name='action_encoding')
    action_spec = action_layers.create_variables()

    lstm_layer = LSTMEncodingNetwork(
      input_tensor_spec=(observation_spec, action_spec),
      preprocessing_layers=joint_preprocessing_layers,
      preprocessing_combiner=joint_preprocessing_combiner,
      conv_layer_params=None,
      input_fc_layer_params=joint_fc_layer_params,
      lstm_size=lstm_size,
      output_fc_layer_params=output_fc_layer_params,
      activation_fn=activation_fn,
      rnn_construction_fn=rnn_construction_fn,
      rnn_construction_kwargs=rnn_construction_kwargs
    )

    value_layer = tf.keras.layers.Dense(
        1,
        activation=None,
        kernel_initializer=last_kernel_initializer,
        name='value'
    )

    super(CriticRnnNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=lstm_layer.state_spec,
        name=name)

    self._observation_layers = observation_layers
    self._action_layers = action_layers
    self._lstm_layer = lstm_layer
    self._value_layer = value_layer

  # TODO(kbanoop): Standardize argument names across different networks.
  def call(self, inputs, step_type, network_state=(), training=False):
    observation, action = inputs
    observation, _ = self._observation_layers(observation, training=training)
    action, _ = self._action_layers(action, training=training)
    state, network_state = self._lstm_layer((observation, action), step_type=step_type, network_state=network_state,
                                            training=training)
    value = self._value_layer(state, training=training)
    return tf.squeeze(value, -1), network_state