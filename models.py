#!/usr/bin/python3

import tensorflow as tf
import tensorflow_gnn as tfgnn
from create_dataset import graph_tensor_spec

class ContinuousFilterConvolution(tf.keras.layers.Layer):
  def __init__(self, units, *, receiver_tag, **kwargs):
    super().__init__(**kwargs)
    self.cutoff = kwargs.get('cutoff', 20.)
    self.gap = kwargs.get('gap', 0.1)
    self.channels = kwargs.get('channels', 256)
    self.shifted_softplus = lambda x: tf.where(x < 14., tf.math.softplus(tf.where(x < 14., x, tf.zeros_like(x))), x) - tf.math.log(2.)
  def build(self, input_shape):
    self.weight1 = self.add_weight(name = 'weight1', shape = (int(tf.math.ceil(self.cutoff / self.gap)), self.channels), initializer = tf.keras.initializers.GlorotUniform(), trainable = True)
    self.bias1 = self.add_weight(name = 'bias1', shape = (self.channels,), initializer = tf.keras.initializers.GlorotUniform(), trainable = True)
    self.weight2 = self.add_weight(name = 'weight2', shape = (self.channels, self.channels), initializer = tf.keras.initializers.GlorotUniform(), trainable = True)
    self.bias2 = self.add_weight(name = 'bias2', shape = (self.channels,), initializer = tf.keras.initializers.GlorotUniform(), trainable = True)
  def call(self, graph, edge_set_name):
    # 1) get weight according to distances between sender and receiver
    receiver_positions = tfgnn.broadcast_node_to_edges(graph, edge_set_name, tfgnn.TARGET, feature_name = "position") # recevier_position.shape = (edge_num, 3)
    sender_positions = tfgnn.broadcast_node_to_edges(graph, edge_set_name, tfgnn.SOURCE, feature_name = "position") # sender_position.shape = (edge_num, 3)
    dists = tf.math.sqrt(tf.math.reduce_sum((receiver_positions - sender_positinos) ** 2, axis = -1, keepdims = True)) # dists.shape = (edge_num, 1)
    centers = tf.expand_dims(tf.linspace(0., self.cutoff, int(tf.math.ceil(self.cutoff / self.gap))), axis = 0) # centers.shape = (1, center_num)
    dists = dists - centers # dists.shape = (edge_num, center_num)
    rbf = tf.math.exp(-(dists ** 2) / self.gap) # rbf.shape = (edge_num, center_num)
    results = self.shifted_softplus(tf.linalg.matmul(rbf, self.weight1) + self.bias1) # results.shape = (edge_num, channels)
    results = self.shifted_softplus(tf.linalg.matmul(results, self.weight2) + self.bias2) # results.shape = (edge_num, channels)
    w = tfgnn.pool_edges_to_node(graph, edge_set_name, tfgnn.TARGET, 'sum', results) # w.shape = (node_num, channels)
    # 2) graph convolution
    receiver_states = tfgnn.broadcast_node_to_edges(graph, edge_set_name, tfgnn.TARGET, feature_name = tfgnn.HIDDEN_STATE) # receiver_states.shape = (edge_num, channels)
    
    # TODO

def SchNet(channels = 200):
  inputs = tf.keras.Input(type_spec = graph_tensor_spec())
  results = inputs.merge_batch_to_components() # merge graphs of a batch to one graph as different components
  results = tfgnn.keras.layers.MapFeatures(
    node_sets_fn = lambda node_set, *, node_set_name: tf.keras.layers.Dense(channels)(node_set[tfgnn.HIDDEN_STATE]),
    edge_sets_fn = lambda edge_set, *, edge_set_name: tf.keras.layers.Dense(channels)(edge_set[tfgnn.HIDDEN_STATE]))(results)
  # only update node vectors
  results = tfgnn.keras.layers.GraphUpdate(
    )
