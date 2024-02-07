#!/usr/bin/python3

from absl import flags, app
from shutil import rmtree
from os import mkdir
from os.path import join, exists
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_csv', default = None, help = 'path to polymer dataset csv')
  flags.DEFINE_string('output_dir', default = 'dataset', help = 'path to output directory')

def smiles_to_sample(smiles, label):
  nodes = list()
  pos = list()
  edges = list()
  molecule = Chem.MolFromSmiles(smiles)
  atom_num = len(molecule.GetAtoms())
  AllChem.EmbedMolecule(molecule)
  AllChem.UFFOptimizeMolecule(molecule)
  for atom in molecule.GetAtoms()
    idx = atom.GetIdx()
    position = molecule.GetConformer().GetAtomPosition(idx)
    nodes.append(atom.GetAtomicNum())
    pos.append((position.x, position.y, position.z))
    for neighbor_atom in atom.GetNeighbors():
      neighbor_idx = neighbor_atom.GetIdx()
      bond = molecule.GetBondBetweenAtoms(idx, neighbor_idx)
      edges.append((idx, neighbor_idx, bond.GetBondType()))
  nodes = tf.stack(nodes, axis = 0) # nodes.shape = (node_num,)
  pos = tf.stack(pos, axis = 0) # pos.shape = (node_num, 3)
  edges = tf.stack(edges, axis = 0) # edges.shape = (edge_num, 3)
  graph = tfgnn.GraphTensor.from_piece(
    node_sets = {
      "atom": tfgnn.NodeSet.from_fields(
        sizes = tf.constant([nodes.shape[0]]),
        features = {
          tfgnn.HIDDEN_STATE: tf.one_hot(nodes, 118),
          "position": pos
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
          tfgnn.HIDDEN_STATE: tf.one_hot(edges[:,2], 22)
        }
      )
    },
    context = tfgnn.Context.from_fields(
      features = {
        "label": tf.constant([label,], dtype = tf.float32)
      }
    )
  )
  return graph

def graph_tensor_spec():
  spec = tfgnn.GraphTensorSpec.from_piece_specs(
      node_sets_spec = {
        "atom": tfgnn.NodeSetSpec.from_field_specs(
          features_spec = {
            tfgnn.HIDDEN_STATE: tf.TensorSpec((None, 118), tf.float32)
            "position": tf.TensorSpec((None, 3), tf.float32)
          },
          sizes_spec = tf.TensorSpec((1,), tf.int32)
        )
      },
      edge_sets_spec = {
        "bond": tfgnn.EdgeSetSpec.from_field_specs(
          features_spec = {
            tfgnn.HIDDEN_STATE: tf.TensorSpec((None, 22), tf.float32)
          },
          sizes_spec = tf.TensorSpec((1,), tf.int32),
          adjacency_spec = tfgnn.AdjacencySpec.from_incident_node_sets("atom", "atom")
        )
      },
      context_spec = tfgnn.ContextSpec.from_field_specs(
        features_spec = {
          'label': tf.TensorSpec(shape = (1,), dtype = tf.float32)
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
  label = context_features.pop('label')
  graph = graph.replace_features(context = context_features)
  return graph, label

def generate_dataset(samples, tfrecord_file):
  writer = tf.io.TFRecordWriter(tfrecord_file)
  for line, (smiles, label) in enumerate(samples):
    graph = smiles_to_sample(smiles, float(label))
    example = tfgnn.write_example(graph)
    writer.write(example.SerializeToString())
  writer.close()

def main(unused_argv):
  if exists(FLAGS.output_dir): rmtree(FLAGS.output_dir)
  mkdir(FLAGS.output_dir)
  samples = list()
  with open(FLAGS.input_csv, 'r') as f:
    for line, row in enumerate(f.readlines()):
      if line == 0: continue
      smiles, label = row.split(',')
      samples.append((smiles, label))
  is_train = np.random.multinomial(1, (9/10,1/10), size = len(samples))[:,0].astype(np.bool_)
  samples = np.array(samples)
  trainset = samples[is_train].tolist()
  valset = samples[np.logical_not(is_train)].tolist()
  generate_dataset(trainset, join(FLAGS.output_dir, 'trainset.tfrecord'))
  generate_dataset(valset, join(FLAGS.output_dir, 'testset.tfrecord'))

if __name__ == "__main__":
  add_options()
  app.run(main)

