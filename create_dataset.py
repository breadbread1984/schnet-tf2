#!/usr/bin/python3

from absl import flags, app
from shutil import rmtree
from os import mkdir, listdir
from os.path import join, exists, splitext, isdir
from math import ceil
from rdkit import Chem
from rdkit.Chem import AllChem
from ase.io.extxyz import read_xyz
from ase.neighborlist import NeighborList
from ase.units import Hartree, eV, Bohr, Ang
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_dir', default = None, help = 'path to qm9 directory')
  flags.DEFINE_string('output_dir', default = 'dataset', help = 'path to output directory')

prop_names = ['rcA', 'rcB', 'rcC', 'mu', 'alpha', 'homo', 'lumo',
              'gap', 'r2', 'zpve', 'energy_U0', 'energy_U', 'enthalpy_H',
              'free_G', 'Cv']
conversions = [1., 1., 1., 1., Bohr ** 3 / Ang ** 3,
               Hartree / eV, Hartree / eV, Hartree / eV,
               Bohr ** 2 / Ang ** 2, Hartree / eV,
               Hartree / eV, Hartree / eV, Hartree / eV,
               Hartree / eV, 1.]

def load_sample(xyz_path, cutoff = 20., gap = 0.1):
  properties = dict()
  # 1) get properties
  with open(xyz_path, 'r') as f:
    lines = f.readlines()
    l = lines[1].split()[2:]
    for pn, p, c in zip(prop_names, l, conversions):
      properties[pn] = float(p) * c
  # 2) get structure
  with open(xyz_path, 'r') as f:
    ats = list(read_xyz(f, 0))[0]
    nbhlist = NeighborList(cutoffs = [cutoff * 0.5] * len(ats), bothways = True, self_interaction = False)
    nbhlist.update(ats)
  nodes = list()
  positions = list()
  edges = list()
  offsets = list()
  cell = ats.cell
  for i in range(len(ats)):
    nodes.append(ats.arrays['numbers'][i])
    positions.append(ats.arrays['positions'][i])
    ind, off = nbhlist.get_neighbors(i)
    sidx = np.argsort(ind)
    ind = ind[sidx]
    off = np.dot(off[sidx], cell) # off.shape = (neighbor_num, 3)
    for j, o in zip(ind, off):
      edges.append((i, j))
      offsets.append(off)
  nodes = tf.stack(nodes, axis = 0) # nodes.shape = (node_num,)
  positions = tf.stack(positions, axis = 0) # pos.shape = (node_num, 3)
  edges = tf.stack(edges, axis = 0) # edges.shape = (edge_num, 2)
  offsets = tf.stack(offsets, axis = 0) # offsets.shape = (edge_num, 3)
  # 3) create graph sample
  graph = tfgnn.GraphTensor.from_pieces(
    node_sets = {
      "atom": tfgnn.NodeSet.from_fields(
        sizes = tf.constant([nodes.shape[0]]),
        features = {
          tfgnn.HIDDEN_STATE: tf.one_hot(nodes, 118),
          "position": positions
        }
      )
    },
    edge_sets = {
      "bond": tfgnn.EdgeSet.from_fields(
        sizes = tf.constant([edges.shape[0]]),
        adjacency = tfgnn.Adjacency.from_indices(
          source = ("atom", edges[:,0]),
          target = ("atom", edges[:,1])
        ),
        features = {
          tfgnn.HIDDEN_STATE: tf.zeros((edges.shape[0], int(ceil(cutoff / gap)))), # rbf
          'offset': offsets,
        }
      )
    },
    context = tfgnn.Context.from_fields(
      features = {
        name: tf.constant([properties[name],], dtype = tf.float32) for name in prop_names
      }
    )
  )
  return graph

def graph_tensor_spec(cutoff = 20, gap = 0.1):
  spec = tfgnn.GraphTensorSpec.from_piece_specs(
    node_sets_spec = {
      "atom": tfgnn.NodeSetSpec.from_field_specs(
        features_spec = {
          tfgnn.HIDDEN_STATE: tf.TensorSpec((None, 118), tf.float32),
          "position": tf.TensorSpec((None, 3), tf.float32)
        },
        sizes_spec = tf.TensorSpec((1,), tf.int32)
      )
    },
    edge_sets_spec = {
      "bond": tfgnn.EdgeSetSpec.from_field_specs(
        features_spec = {
          tfgnn.HIDDEN_STATE: tf.TensorSpec((None, int(ceil(cutoff / gap))), dtype = tf.float32), # rbf
          'offset': tf.TensorSpec((None, 3), tf.float32),
        },
        sizes_spec = tf.TensorSpec((1,), tf.int32),
        adjacency_spec = tfgnn.AdjacencySpec.from_incident_node_sets("atom", "atom")
      )
    },
    context_spec = tfgnn.ContextSpec.from_field_specs(
      features_spec = {
        name: tf.TensorSpec(shape = (1,), dtype = tf.float32) for name in prop_names
      }
    )
  )
  return spec

def parse_function(serialized_example):
  graph = tfgnn.parse_single_example(
    graph_tensor_spec(),
    serialized_example,
    validate = True)
  context_features = graph.context.get_features_dict()
  labels = {name: context_features.pop(name) for name in prop_names}
  graph = graph.replace_features(context = context_features)
  return graph, labels

def generate_dataset(samples, tfrecord_file):
  writer = tf.io.TFRecordWriter(tfrecord_file)
  for line, xyz_path in enumerate(samples):
    # convert file content
    with open(xyz_path, 'r') as f:
      lines = f.readlines()
      with open('tmp.xyz', 'w') as fout:
        for line in lines:
          fout.write(line.replace('*^', 'e'))
    # write to tfrecord
    graph = load_sample('tmp.xyz')
    example = tfgnn.write_example(graph)
    writer.write(example.SerializeToString())
  writer.close()

def main(unused_argv):
  if exists(FLAGS.output_dir): rmtree(FLAGS.output_dir)
  mkdir(FLAGS.output_dir)
  samples = list()
  for f in listdir(FLAGS.input_dir):
    stem, ext = splitext(f)
    if ext != '.xyz' or isdir(join(FLAGS.input_dir, f)): continue
    samples.append(join(FLAGS.input_dir, f))
  is_train = np.random.multinomial(1, (9/10,1/10), size = len(samples))[:,0].astype(np.bool_)
  samples = np.array(samples)
  trainset = samples[is_train].tolist()
  valset = samples[np.logical_not(is_train)].tolist()
  generate_dataset(trainset, join(FLAGS.output_dir, 'trainset.tfrecord'))
  generate_dataset(valset, join(FLAGS.output_dir, 'testset.tfrecord'))

if __name__ == "__main__":
  add_options()
  app.run(main)

