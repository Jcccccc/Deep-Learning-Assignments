from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from lib.layer_utils import *


""" Super Class """
class Module(object):
    def __init__(self):
        self.params = {}
        self.grads = {}

    def forward(self, feat, is_training=True, seed=None):
        output = feat
        for layer in self.net.layers:
            if isinstance(layer, dropout):
                output = layer.forward(output, is_training, seed)
            else:
                output = layer.forward(output)
        self.net.gather_params()
        return output

    def backward(self, dprev):
        for layer in self.net.layers[::-1]:
            dprev = layer.backward(dprev)
        self.net.gather_grads()
        return dprev


""" Classes """
class TestFCReLU(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            flatten(name="flat"),
            fc(input_dim=20, output_dim=10, name="fc"),
            relu(name="relu")
            ########### END ###########
        )


class SmallFullyConnectedNetwork(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            flatten(name="flat"),
            fc(input_dim=4, output_dim=30, name="fc1"),
            relu(name="relu1"),
            fc(input_dim=30, output_dim=7, name="fc2"),
            relu(name="relu2")
            ########### END ###########
        )


class DropoutNet(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        self.dropout = dropout
        self.seed = seed
        self.net = sequential(
            flatten(name="flat"),
            fc(15, 20, 5e-2, name="fc1"),
            relu(name="relu1"),
            fc(20, 30, 5e-2, name="fc2"),
            relu(name="relu2"),
            fc(30, 10, 5e-2, name="fc3"),
            relu(name="relu3"),
            dropout(keep_prob, seed=seed)
        )


class TinyNet(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        """ Some comments """
        self.net = sequential(
            ########## TODO: ##########
            flatten(name="flat"),
            fc(input_dim=3072, output_dim=512, name="fc1"),
            relu(name="relu1"),
            fc(input_dim=512, output_dim=10, name="fc2")
            ########### END ###########
        )


class DropoutNetTest(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        """ Some comments """
        self.dropout = dropout
        self.seed = seed
        self.net = sequential(
            flatten(name="flat"),
            fc(3072, 500, 1e-2, name="fc1"),
            dropout(keep_prob, seed=seed),
            relu(name="relu1"),
            fc(500, 500, 1e-2, name="fc2"),
            relu(name="relu2"),
            fc(500, 10, 1e-2, name="fc3"),
        )


class FullyConnectedNetwork_2Layers(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        """ Some comments """
        self.net = sequential(
            flatten(name="flat"),
            fc(5, 5, name="fc1"),
            relu(name="relu1"),
            fc(5, 5, name="fc2")
        )


class FullyConnectedNetwork(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        """ Some comments """
        self.net = sequential(
            flatten(name="flat"),
            fc(3072, 100, 5e-2, name="fc1"),
            relu(name="relu1"),
            fc(100, 100, 5e-2, name="fc2"),
            relu(name="relu2"),
            fc(100, 100, 5e-2, name="fc3"),
            relu(name="relu3"),
            fc(100, 100, 5e-2, name="fc4"),
            relu(name="relu4"),
            fc(100, 100, 5e-2, name="fc5"),
            relu(name="relu5"),
            fc(100, 10, 5e-2, name="fc6")
        )

