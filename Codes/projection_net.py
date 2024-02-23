# !/usr/bin/env python3
#  -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 15:21:46 2017

@author: shubham
"""
import tensorflow as tf
from net import Net
from cost_functions import MAE


class ProjectionNet(Net):
    def __init__(self, arg_dict={}):
        # self.print("Initializing vars:")
        super().__init__(arg_dict)
        # DEFAULT VALUES
        self.cost_function = MAE
        self.print("="*25)
        self.input_shape = [None,256,256,256]
        self.output_shape = [None,1,1,1]
        self._init_all(arg_dict)

    def _build_network(self, sess, cost_function, cost_function_name):
        """Method to create the graph."""
        with tf.variable_scope(self.scope):
            input_layer = tf.placeholder(dtype=tf.float32, shape = self.input_shape)
            y_true = tf.placeholder(dtype=tf.float32, shape=self.output_shape)
            self.inputs.append(input_layer)
            self.true_outputs.append(y_true)
            block_1 = self._conv_bn_relu_block("block1", input_layer,
                                               [3, 3, 256, 128])
            block_2 = self._conv_bn_relu_block("block2", block_1,
                                               [3, 3, 128, 64])
            block_3 = self._conv_bn_relu_block("block3", block_2,
                                               [3, 3, 64, 32])
            block_4 = self._conv_bn_relu_block("block4", block_3,
                                               [3, 3, 32, 16])
            block_5 = self._conv_bn_relu_block("block5", block_4,
                                               [3, 3, 16, 8])
            block_6 = self._conv_bn_block("block6", block_5,
                                          [3, 3, 8, 4])
            # Classification Part
            block_7 = self._conv_bn_relu_pool_block("block7", block_6,
                                                    [3, 3, 4, 16])
            block_8 = self._conv_bn_relu_pool_block("block8", block_7,
                                                    [3, 3, 16, 32])
            block_9 = self._conv_bn_relu_pool_block("block9", block_8,
                                                    [3, 3, 32, 64])
            block_10 = self._conv_bn_relu_pool_block("block10", block_9,
                                                     [3, 3, 64, 128])
            block_11 = self._conv_bn_relu_pool_block("block11", block_10,
                                                     [3, 3, 128, 256], [4, 4],
                                                     [4, 4])
            self.print(block_11.get_shape())
            block_12 = self._conv_bn_relu_pool_block("block12", block_11,
                                                   [3, 3, 256, 512], [4, 4],
                                                   [1, 1])
            y_pred = self._conv_block("y_pred", block_12,
                                      [1, 1, 512, 1], padding="VALID")
            self.outputs.append(y_pred)
            self.print(y_pred.get_shape())
            with tf.variable_scope(cost_function_name):
                cost = cost_function(y_true, y_pred)
            self.costs.append({cost_function_name:cost})
            self.print("Network Built")
        with sess.as_default():
            sess.run(tf.global_variables_initializer())
            self._built_flag = True

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
            self.histogram_list.append(tf.summary.histogram("W", W))  # for summary
            conv = tf.nn.conv2d(input_layer, W, [1, 1, 1, 1], padding="SAME",
                                name="conv")
            bias = self._get_trainable_tensor("bias", [k_shape[3]])
            self.histogram_list.append(tf.summary.histogram("b", bias))
            conv = tf.nn.bias_add(conv, bias)
            bn = tf.layers.batch_normalization(conv)
        return bn

    def _conv_block(self, block_name, input_layer, k_shape, padding = "SAME"):
        with tf.variable_scope(block_name):
            W = self._get_trainable_tensor("W", k_shape)
            self.histogram_list.append(tf.summary.histogram("W", W))  # for summary
            conv = tf.nn.conv2d(input_layer, W, [1, 1, 1, 1], padding=padding,
                                name="conv")
            bias = self._get_trainable_tensor("bias", [k_shape[3]])
            self.histogram_list.append(tf.summary.histogram("b", bias))
            conv = tf.nn.bias_add(conv, bias)
        return conv

    def _conv_bn_relu_block(self, block_name, input_layer, k_shape):
        with tf.variable_scope(block_name):
            W = self._get_trainable_tensor("W", k_shape)
            self.histogram_list.append(tf.summary.histogram("W", W))  # for summary
            conv = tf.nn.conv2d(input_layer, W, [1, 1, 1, 1], padding="SAME",
                                name="conv")
            bias = self._get_trainable_tensor("bias", [k_shape[3]])
            self.histogram_list.append(tf.summary.histogram("b", bias))
            conv = tf.nn.bias_add(conv, bias)
            bn = tf.layers.batch_normalization(conv)
            relu = tf.nn.relu(bn, name="relu")
        return relu

    def _conv_bn_relu_pool_block(self, block_name, input_layer, k_shape,
                                 pool_shape=[2, 2], pool_stride=[2, 2]):
        with tf.variable_scope(block_name):
            W = self._get_trainable_tensor("W", k_shape)
            self.histogram_list.append(tf.summary.histogram("W", W))  # for summary
            conv = tf.nn.conv2d(input_layer, W, [1, 1, 1, 1], padding="SAME",
                                name="conv")
            bias = self._get_trainable_tensor("bias", [k_shape[3]])
            self.histogram_list.append(tf.summary.histogram("b", bias))
            conv = tf.nn.bias_add(conv, bias)
            bn = tf.layers.batch_normalization(conv)
            relu = tf.nn.relu(bn, name="relu")
            pool = tf.layers.max_pooling2d(relu, pool_shape, pool_stride,
                                           padding="VALID", name="max_pool")
        return pool

    def _get_trainable_tensor(self, name, shape):
        """ shape must be of the form: [filter_depth, filter_height,
        filter_width, in_channels, out_channels](for 3D conv) or
        [filter_height, filter_width, in_channels, out_channels](for 2D conv)
        """
        return tf.get_variable(name=name, shape=shape,
                               initializer=tf.contrib.layers.xavier_initializer()
                               )


if __name__ == '__main__':
    import os
    pnet = ProjectionNet()
    pnet.build_network(tf.Session(), MAE, "M.Abs.Err.")
    g = tf.summary.merge_all()
    f_wtr = tf.summary.FileWriter(os.getcwd()+"/summary_file")
    with tf.Session() as sess:
        f_wtr.add_graph(sess.graph)
