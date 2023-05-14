"""
Copied from agents/tf_agents/agents/ddpg/critic_network.py
TODO(samarth): remove this file once upstream agents includes EncodingNetworks
"""

import tensorflow as tf
from tf_agents.networks.utils import mlp_layers
from tf_agents.networks.network import Network
from tf_agents.networks.encoding_network import EncodingNetwork


class CriticNetwork(Network):
  def __init__(self,
               input_tensor_spec,
               observation_preprocessing_layers=None,
               observation_preprocessing_combiner=None,
               observation_conv_layer_params=None,
               observation_fc_layer_params=None,
               observation_dropout_layer_params=None,
               action_preprocessing_layers=None,
               action_preprocessing_combiner=None,
               action_fc_layer_params=None,
               action_dropout_layer_params=None,
               joint_fc_layer_params=None,
               joint_dropout_layer_params=None,
               activation_fn=tf.nn.relu,
               output_activation_fn=None,
               kernel_initializer=None,
               last_kernel_initializer=None,
               name='CriticNetwork'):
    super(CriticNetwork, self).__init__(input_tensor_spec=input_tensor_spec, state_spec=(), name=name)

    observation_spec, action_spec = input_tensor_spec

    if kernel_initializer is None:
      kernel_initializer = tf.compat.v1.keras.initializers.VarianceScaling(
          scale=1. / 3., mode='fan_in', distribution='uniform')
    if last_kernel_initializer is None:
      last_kernel_initializer = tf.keras.initializers.RandomUniform(
          minval=-0.003, maxval=0.003)

    self._observation_net = EncodingNetwork(observation_spec, observation_preprocessing_layers,
                                            observation_preprocessing_combiner, observation_conv_layer_params,
                                            observation_fc_layer_params, observation_dropout_layer_params,
                                            activation_fn, kernel_initializer=kernel_initializer,
                                            name='observation_encoding')

    self._action_net = EncodingNetwork(action_spec, action_preprocessing_layers, action_preprocessing_combiner,
                                       fc_layer_params=action_fc_layer_params,
                                       dropout_layer_params=action_dropout_layer_params, activation_fn=activation_fn,
                                       kernel_initializer=kernel_initializer, name='action_encoding')

    self._joint_layers = mlp_layers(None, joint_fc_layer_params, joint_dropout_layer_params,
                                    activation_fn=activation_fn, kernel_initializer=kernel_initializer,
                                    name='joint_mlp')

    self._joint_layers.append(tf.keras.layers.Dense(1, activation=output_activation_fn,
                                                    kernel_initializer=last_kernel_initializer, name='value'))
  
  
  def call(self, inputs, step_type=(), network_state=(), training=False):
    observations, actions = inputs
    del step_type
    observations, _ = self._observation_net(observations)
    actions, _ = self._action_net(actions)
    joint = tf.concat([observations, actions], 1)
    for layer in self._joint_layers:
      joint = layer(joint, training=training)
    return tf.reshape(joint, [-1]), network_state