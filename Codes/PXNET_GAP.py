#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 17:47
Using PXNET of exp 13 as base. PXNET_with_last layer with GAP.
@author: shubham
"""
import tensorflow as tf
import logger
import data
from net import Net
import numpy as np
import utility as ut
from cost_functions import MSE, XENT

def get_cost_fn (name):
    if name=="XENT":
        return XENT
    elif name=="MSE":
        return MSE
    else:
        raise ValueError()

def one_hot (input_):
    with tf.variable_scope("one_hot"):
        return tf.one_hot(tf.argmax(input_,1), tf.shape(input_)[1])

def create_accuracy_node(y_true, y_pred_oh):
    with tf.variable_scope("ACCURACY"):
        sp = y_true.get_shape()
        mm = tf.multiply(y_true, y_pred_oh)
        acc = tf.reduce_sum(mm,axis=1)
        acc = tf.reduce_mean(acc)
    return acc

class ProjectionNet(Net):
    def __init__(self,arg_dict={}):
        super().__init__({})
        self.scope = "PXNet"
        self.node_dict = {}
        self.labels = self.true_outputs# alias
        self.log_list = self.histogram_list# alias
        self.initializer = tf.contrib.layers.xavier_initializer(seed=19)
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=1e-5)
        self.output_shape=(None,3)
        self.input_shape=(None,95,69,79)
        self.compression_part_var_list=None
        self.compression_part_saver = None
        self.classification_part_var_list = None
        self.cost_function = "XENT"
        self._init_all(arg_dict)


    def _build_network(self):
        with tf.variable_scope(self.scope):
            input_layer = tf.placeholder(dtype=tf.float32, shape = self.input_shape)
            y_true = tf.placeholder(dtype=tf.float32, shape=self.output_shape)
            self.inputs.append(input_layer)
            dims = input_layer.get_shape()
            ch = int(dims[3])
            block_1 = self._conv_bn_relu_block("block1", input_layer,
                                               [3, 3, ch, (int)(ch/2.0)])
            print ("block_1:",block_1.get_shape())
            block_2 = self._conv_bn_relu_block("block2", block_1,
                                               [3, 3, (int)(ch/2.0), (int)(ch/4.0)])
            print ("block_2:",block_2.get_shape())
            block_3 = self._conv_bn_relu_block("block3", block_2,
                                               [3, 3, (int)(ch/4.0), (int)(ch/8.0)])
            print ("block_3:",block_3.get_shape())
            block_4 = self._conv_bn_relu_block("block4", block_3,
                                               [3, 3, (int)(ch/8.0), (int)(ch/16.0)])
            print ("block_4:",block_4.get_shape())
            block_5 = self._conv_bn_sigmoid_block("block5", block_4,
                                               [3, 3, (int)(ch/16.0), 1])# gedding 2d
            print ("block_5:",block_5.get_shape())
            self.extra_outputs["projection"] = block_5

            self.compression_part_var_list = (self.get_var_list()).copy()
            self.compression_part_saver = tf.train.Saver(self.compression_part_var_list)


            # Classification Part
            block_6 = self._conv_bn_relu_pool_block("block6", block_5,
                                                    [3, 3, 1, 4])
            print ("block_6:",block_6.get_shape())
            block_7 = self._conv_bn_relu_pool_block("block7", block_6,
                                                    [3, 3, 4, 8])
            print ("block_7:",block_7.get_shape())
            block_8 = self._conv_bn_relu_pool_block("block8", block_7,
                                                    [3, 3, 8, 16])
            print ("block_8:",block_8.get_shape())
            block_9 = self._conv_bn_relu_pool_block("block9", block_8,
                                                     [3, 3, 16, 32])
            print ("block_9:",block_9.get_shape())
            
            block_10 = self._conv_bn_relu_GAP_block("block10",block_9,
                                                    [3, 3, 32, 64])
            
            print ("block_10:",block_10.get_shape())
            block_11 = tf.layers.conv2d(block_10,3,[1,1])
            self.print("block_11",block_11.get_shape())
            flattened = tf.reshape(block_11,[-1,3]) #Flatten(name="flatten_1")(block_12)
            self.print("flattened:",flattened.get_shape())
            y_pred = tf.nn.softmax(flattened,name="softmax")
            self.print("y_pred:",y_pred.get_shape())
            
            cost_fn = get_cost_fn(self.cost_function)
            cost = cost_fn(y_true, y_pred)

            max_y_pred = one_hot(y_pred)
            self.outputs.append(y_pred)
            self.outputs.append(max_y_pred)
            self.print("CALCULATED OUTPUT SHAPE:",max_y_pred.get_shape())

            self.costs.append(cost)
            self.accuracy = create_accuracy_node(y_true, max_y_pred)
            self.classification_part_var_list = ut.get_list_difference(
                    self.get_var_list(),
                    self.compression_part_var_list)


            self.outputs.append(block_5)
            self.true_outputs.append(y_true)
            self.costs.append(cost)
            self.node_dict[self.cost_function] = cost
            self.node_dict["Accuracy"] = self.accuracy
            print("===Compression Part Var List===")
            ut.print_sequence(self.compression_part_var_list)
            print("===Classification Part Var List===")
            ut.print_sequence(self.classification_part_var_list)

    def build_network(self):
        """Should be explicitly called in a session to initialize the graph weig-
        hts or load it from the disk
        """
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._build_network()
            self.saver = tf.train.Saver(self.get_var_list(),max_to_keep=None)
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
    
    def _conv_bn_relu_GAP_block(self, block_name, input_layer, k_shape):
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
            pool = tf.reduce_mean(relu, axis=[1,2],name="GAP",keep_dims=True)
        return pool
    
    def _conv_bn_sigmoid_block(self, block_name, input_layer, k_shape):
        with tf.variable_scope(block_name):
            W = self._get_trainable_tensor("W", k_shape)
            self.log_list.append(tf.summary.histogram("W", W))  # for summary
            conv = tf.nn.conv2d(input_layer, W, [1, 1, 1, 1], padding="SAME",
                                name="conv")
            bias = self._get_trainable_tensor("bias", [k_shape[3]])
            self.log_list.append(tf.summary.histogram("b", bias))
            conv = tf.nn.bias_add(conv, bias)
            bn = tf.layers.batch_normalization(conv)
            sig = tf.nn.sigmoid( bn, name="sigmoid")
        return sig

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
    model = ProjectionNet()
    model.build_network()
