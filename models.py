#!/usr/bin/python3

import tensorflow as tf
import tensorflow_gnn as tfgnn
from create_dataset import graph_tensor_spec, prop_names

shifted_softplus = lambda x: tf.where(x < 14., tf.math.softplus(tf.where(x < 14., x, tf.zeros_like(x))), x) - tf.math.log(2.)

class FilterNet(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(FilterNet, self).__init__()
    self.cutoff = kwargs.get('cutoff', 20.)
    self.gap = kwargs.get('gap', 0.1)
  def call(self, inputs):
    edge_features, node_features, context_features = inputs
    # 1) filternet
    sender_positions = node_features[tfgnn.SOURCE] # sender_position.shape = (edge_num, 3)
    receiver_positions = node_features[tfgnn.TARGET] # receiver_positions.shape = (edge_num, 3)
    # 1) filternet
    offsets = edge_features # offsets.shape = (edge_num, 3)
    dists = tf.math.sqrt(tf.math.reduce_sum((sender_positions - (receiver_positions + offsets)) ** 2, axis = -1, keepdims = True)) # dists.shape = (edge_num, 1)
    centers = tf.expand_dims(tf.linspace(0., self.cutoff, int(tf.math.ceil(self.cutoff / self.gap))), axis = 0) # centers.shape = (1, center_num)
    dists = dists - centers # dists.shape = (edge_num, center_num)
    rbf = tf.math.exp(-(dists ** 2) / self.gap) # rbf.shape = (edge_num, center_num)
    return rbf
  def get_config(self):
    config = super(FilterNet, self).get_config()
    config['cutoff'] = self.cutoff
    config['gap'] = self.gap
    return config
  @classmethod
  def from_config(cls, config):
    return cls(**config)

class ContinuousFilterConvolution(tf.keras.layers.Layer):
  def __init__(self, units, **kwargs):
    super(ContinuousFilterConvolution, self).__init__(**kwargs)
    self.channels = units
    self.cutoff = kwargs.get('cutoff', 20.)
    self.gap = kwargs.get('gap', 0.1)
  def build(self, input_shape):
    self.weight1 = self.add_weight(name = 'weight1', shape = (int(tf.math.ceil(self.cutoff / self.gap)), self.channels), initializer = tf.keras.initializers.GlorotUniform(), trainable = True)
    self.bias1 = self.add_weight(name = 'bias1', shape = (self.channels,), initializer = tf.keras.initializers.GlorotUniform(), trainable = True)
    self.weight2 = self.add_weight(name = 'weight2', shape = (self.channels, self.channels), initializer = tf.keras.initializers.GlorotUniform(), trainable = True)
    self.bias2 = self.add_weight(name = 'bias2', shape = (self.channels,), initializer = tf.keras.initializers.GlorotUniform(), trainable = True)
    self.weight3 = self.add_weight(name = 'weight3', shape = (self.channels, self.channels), initializer = tf.keras.initializers.GlorotUniform(), trainable = True)
    # NOTE: dense3 doesnt use bias, so no bias3
    self.weight4 = self.add_weight(name = 'weight4', shape = (self.channels, self.channels), initializer = tf.keras.initializers.GlorotUniform(), trainable = True)
    self.bias4 = self.add_weight(name = 'bias4', shape = (self.channels,), initializer = tf.keras.initializers.GlorotUniform(), trainable = True)
    self.weight5 = self.add_weight(name = 'weight5', shape = (self.channels, self.channels), initializer = tf.keras.initializers.GlorotUniform(), trainable = True)
    self.bias5 = self.add_weight(name = 'bias5', shape = (self.channels,), initializer = tf.keras.initializers.GlorotUniform(), trainable = True)
  def call(self, graph, edge_set_name):
    rbf = tfgnn.keras.layers.Readout(edge_set_name = edge_set_name, feature_name = 'rbf')(graph) # rbf.shape = (edge_num, center_num)
    results = shifted_softplus(tf.linalg.matmul(rbf, self.weight1) + self.bias1) # results.shape = (edge_num, channels)
    w = shifted_softplus(tf.linalg.matmul(results, self.weight2) + self.bias2) # results.shape = (edge_num, channels)
    # 2) continuous fileter convolution
    x = tfgnn.keras.layers.Readout(node_set_name = 'atom', feature_name = tfgnn.HIDDEN_STATE)(graph) # x.shape = (node_num, channels)
    f = tf.linalg.matmul(x, self.weight3) # results.shape = (node_num, channels)
    f = tfgnn.broadcast_node_to_edges(graph, edge_set_name, tfgnn.SOURCE, feature_value = f) # receiver_states.shape = (edge_num, channels)
    wf = w * f # wf.shape = (node_num, channels)
    conv = tfgnn.pool_edges_to_node(graph, edge_set_name, tfgnn.SOURCE, reduce_type = 'sum', feature_value = wf) # conv.shape = (node_num, channels)
    y = shifted_softplus(tf.linalg.matmul(conv, self.weight4) + self.bias4) # y.shape = (node_num, channels)
    # 3) dense
    v = tf.linalg.matmul(y, self.weight5) + self.bias5 # v.shape = (node_num, channels)
    y = x + v
    return y
  def get_config(self):
    config = super(ContinuousFilterConvolution, self).get_config()
    config['channels'] = self.channels
    config['cutoff'] = self.cutoff
    config['gap'] = self.gap
    return config
  @classmethod
  def from_config(cls, config):
    return cls(**config)

class UpdateNodeHidden(tf.keras.layers.Layer):
  def __init__():
    super(UpdateNodeHidden, self).__init__()
  def call(self, inputs):
    node_features, incident_node_features, context_features = inputs
    return incident_node_features

def SchNet(channels = 256, layer_num = 4):
  graph = tf.keras.Input(type_spec = graph_tensor_spec())
  graph = graph.merge_batch_to_components() # merge graphs of a batch to one graph as different components
  graph = tfgnn.keras.layers.MapFeatures(
    node_sets_fn = lambda node_set, *, node_set_name: {  
      tfgnn.HIDDEN_STATE: tf.keras.layers.Dense(channels)(node_set[tfgnn.HIDDEN_STATE]),
      'position': node_set['position']
    }
  )(graph)
  # update edge feature rbf once
  graph = tfgnn.keras.layers.GraphUpdate(
    edge_sets = {
      "bond": tfgnn.keras.layers.EdgeSetUpdate(
        edge_input_feature = "offset",
        node_input_feature = "position",
        node_input_tags = [tfgnn.SOURCE, tfgnn.TARGET],
        next_state = FilterNet()
      )
    }
  )(graph)
  # update node
  for i in range(layer_num):
    graph = tfgnn.keras.layers.GraphUpdate(
      node_sets = {
        "atom": tfgnn.keras.layers.NodeSetUpdate(
          edge_set_inputs = {
            "bond": ContinuousFilterConvolution(units = channels)
          },
          next_state = UpdateNodeHidden()
        )
      }
    )(graph)
  resutls = tfgnn.keras.layers.Pool(tag = tfgnn.CONTEXT, reduce_type = "mean", node_set_name = "atom")(graph)
  results = tf.keras.layers.Dense(channels // 2, activation = shifted_softplus)(results) # results.shape = (batch, channels // 2)
  results = tf.keras.layers.Dense(len(prop_names))(results) # results.shape = (batch, len(prop_names))
  results = tf.keras.layers.Lambda(lambda x, n: tf.split(x, n, axis = -1), arguments = {'n': len(prop_names)})(results)
  return tf.keras.Model(inputs = graph, outputs = {k:v for k,v in zip(prop_names,results)})
