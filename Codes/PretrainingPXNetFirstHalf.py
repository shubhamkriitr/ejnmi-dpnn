#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 21:41:38 2018
For Pretraining: Parts of Projection Net has been implemented here.
@author: shubham
"""
import tensorflow as tf
import logger
import data
from net import Net
import numpy as np
import utility as utl
from cost_functions import MSE, SSIM_UK

class ProjectionNetFirstHalf(Net):
    def __init__(self,arg_dict={}):
        super().__init__({})
        self.scope = "PXNet"
        self.node_dict = {}
        self.labels = self.true_outputs# alias
        self.log_list = self.histogram_list# alias
        self.initializer = tf.contrib.layers.xavier_initializer(seed=19)
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=1e-5)
        self.output_shape=(None,95,69,1)
        self.input_shape=(None,95,69,79)
        self._init_all(arg_dict)


    def _build_network(self):
        with tf.variable_scope(self.scope):
            input_layer = tf.placeholder(dtype=tf.float32, shape = self.input_shape)
            self.inputs.append(input_layer)
            dims = input_layer.get_shape()
            ch = int(dims[3])
            block_1 = self._conv_bn_relu_block("block1", input_layer,
                                               [3, 3, ch, (int)(ch/2.0)])
            block_2 = self._conv_bn_relu_block("block2", block_1,
                                               [3, 3, (int)(ch/2.0), (int)(ch/4.0)])
            block_3 = self._conv_bn_relu_block("block3", block_2,
                                               [3, 3, (int)(ch/4.0), (int)(ch/8.0)])
            block_4 = self._conv_bn_relu_block("block4", block_3,
                                               [3, 3, (int)(ch/8.0), (int)(ch/16.0)])
            block_5 = self._conv_bn_relu_block("block5", block_4,
                                               [3, 3, (int)(ch/16.0), 1])# gedding 2d
            print ("block_5:",block_5.get_shape())
            y_true = tf.placeholder(dtype=tf.float32, shape=self.output_shape)
            ssim_loss = -SSIM_UK(y_true,block_5)
            l2_loss = MSE(y_true,block_5)
            cost = l2_loss + ssim_loss
            self.outputs.append(block_5)
            self.true_outputs.append(y_true)
            self.costs.append(cost)
            self.node_dict["MSE"] = l2_loss
            self.node_dict["SSIM"] = ssim_loss
            self.node_dict["LOSS"] = cost

    def build_network(self):
        """Should be explicitly called in a session to initialize the graph weig-
        hts or load it from the disk
        """
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._build_network()
            self.saver = tf.train.Saver( self.get_var_list(),max_to_keep=None)
        self._built_flag = True
        self.num_of_inputs = len(self.inputs)

    def get_var_list(self):
        return self.graph.get_collection("variables")

    def _conv_bn_block(self, block_name, input_layer, k_shape):
        """Returns a tensor-node after adding conv-bn-relu operations
        after the given input_layer.

        Args:
            block_name:
            input_layer:
            k_shape: A sequence of four numbers
        """
        with tf.variable_scope(block_name):
            W = self._get_trainable_tensor("W", k_shape)
            self.log_list.append(tf.summary.histogram("W", W))  # for summary
            conv = tf.nn.conv2d(input_layer, W, [1, 1, 1, 1], padding="SAME",
                                name="conv")
            bias = self._get_trainable_tensor("bias", [k_shape[3]])
            self.log_list.append(tf.summary.histogram("b", bias))
            conv = tf.nn.bias_add(conv, bias)
            bn = tf.layers.batch_normalization(conv)
        return bn

    def _conv_block(self, block_name, input_layer, k_shape, padding = "SAME"):
        with tf.variable_scope(block_name):
            W = self._get_trainable_tensor("W", k_shape)
            self.log_list.append(tf.summary.histogram("W", W))  # for summary
            conv = tf.nn.conv2d(input_layer, W, [1, 1, 1, 1], padding=padding,
                                name="conv")
            bias = self._get_trainable_tensor("bias", [k_shape[3]])
            self.log_list.append(tf.summary.histogram("b", bias))
            conv = tf.nn.bias_add(conv, bias)
        return conv

    def _conv_bn_relu_block(self, block_name, input_layer, k_shape):
        with tf.variable_scope(block_name):
            W = self._get_trainable_tensor("W", k_shape)
            self.log_list.append(tf.summary.histogram("W", W))  # for summary
            conv = tf.nn.conv2d(input_layer, W, [1, 1, 1, 1], padding="SAME",
                                name="conv")
            bias = self._get_trainable_tensor("bias", [k_shape[3]])
            self.log_list.append(tf.summary.histogram("b", bias))
            conv = tf.nn.bias_add(conv, bias)
            bn = tf.layers.batch_normalization(conv)
            relu = tf.nn.relu(bn, name="relu")
        return relu

    def _conv_bn_relu_pool_block(self, block_name, input_layer, k_shape,
                                 pool_shape=[2, 2], pool_stride=[2, 2]):
        with tf.variable_scope(block_name):
            W = self._get_trainable_tensor("W", k_shape)
            self.log_list.append(tf.summary.histogram("W", W))  # for summary
            conv = tf.nn.conv2d(input_layer, W, [1, 1, 1, 1], padding="SAME",
                                name="conv")
            bias = self._get_trainable_tensor("bias", [k_shape[3]])
            self.log_list.append(tf.summary.histogram("b", bias))
            conv = tf.nn.bias_add(conv, bias)
            bn = tf.layers.batch_normalization(conv)
            relu = tf.nn.relu(bn, name="relu")
            pool = tf.layers.max_pooling2d(relu, pool_shape, pool_stride,
                                           padding="VALID", name="max_pool")
        return pool

    def fc_block(self, in_layer, name):
        with tf.variable_scope("fc_block_{}".format(name)):
            flattened = Flatten(name="flatten_1")(in_layer)
            out = dense(flattened, 3, activation=None)# Linear Activation
        return out

    def _get_trainable_tensor(self, name, shape):
        """ shape must be of the form: [filter_depth, filter_height,
        filter_width, in_channels, out_channels](for 3D conv) or
        [filter_height, filter_width, in_channels, out_channels](for 2D conv)
        """
        return tf.get_variable(name=name, shape=shape,
                               initializer = self.initializer,
                               regularizer=self.regularizer)


if __name__ == '__main__':
    model = ProjectionNetFirstHalf()
    model.build_network()
