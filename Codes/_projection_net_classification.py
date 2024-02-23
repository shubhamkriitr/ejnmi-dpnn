# !/usr/bin/env python3
#  -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 15:21:46 2017

@author: shubham
"""
import tensorflow as tf
from tensorflow.python.layers.core import Flatten, dense
from _net import Net
def MAE (y_true, y_pred, scope = "MAE"):
    with tf.variable_scope(scope):
        cost = tf.reduce_mean(tf.losses.absolute_difference(y_true, y_pred))
    return cost

def MSE (y_true, y_pred, scope="MSE"):
    with tf.variable_scope(scope):
        cost = tf.losses.mean_squared_error(y_true, y_pred)
    return cost

def XENT  (y_true, y_pred, scope="CENT"):
    with tf.variable_scope(scope):
        y_pred = tf.clip_by_value(y_pred,1e-7,1.0)
        cost = - tf.reduce_mean(y_true*tf.log(y_pred) +
                                (1-y_true)*tf.log(1-y_pred+1e-7))
    return cost

cost_functions = {"MAE":MAE, "MSE":MSE, "XENT":XENT}

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
    def __init__(self, arg_dict={}):
        # self.print("Initializing vars:")
        super().__init__(arg_dict)
        # DEFAULT VALUES
        self.cost_function = MAE
        self.weight_decay = None
        self.print("="*25)
        self.input_shape = None
        self.output_shape = [None,3]
        self._init_all(arg_dict)

    def _build_network(self, sess):
        """Method to create the graph."""
        with tf.variable_scope(self.scope):
            input_layer = tf.placeholder(dtype=tf.float32, shape = self.input_shape)
            y_true = tf.placeholder(dtype=tf.float32, shape=self.output_shape)
            self.inputs.append(input_layer)
            self.labels.append(y_true)
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
            block_6 = block_5
            # Classification Part
            block_7 = self._conv_bn_relu_pool_block("block7", block_6,
                                                    [3, 3, 1, 4])
            block_8 = self._conv_bn_relu_pool_block("block8", block_7,
                                                    [3, 3, 4, 8])
            block_9 = self._conv_bn_relu_pool_block("block9", block_8,
                                                    [3, 3, 8, 16])
            block_10 = self._conv_bn_relu_pool_block("block10", block_9,
                                                     [3, 3, 16, 32])
            block_11 = self._conv_bn_relu_pool_block("block11", block_10,
                                                     [3, 3, 32, 64], [4, 4],
                                                     [4, 4])
            self.print(block_11.get_shape())
            block_12 =self.fc_block(block_11,"1")
            self.print(block_12.get_shape())
            
            y_pred = tf.nn.softmax(block_12,name="softmax")
            cost = cost_functions[self.cost_function](y_true, y_pred)
            max_y_pred = one_hot(y_pred)
            self.outputs.append(y_pred)
            self.outputs.append(max_y_pred)
            self.print("CALCULATED OUTPUT SHAPE:",max_y_pred.get_shape())
            #cost = MSE(y_true, y_pred, "MSE")
            self.costs.append(cost)
            self.costs.append(create_accuracy_node(y_true, max_y_pred))
            self.print("Network Built")
        with sess.as_default():
            sess.run(tf.global_variables_initializer())
            self._built_flag = True
        self.saver = tf.train.Saver()

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
        reg = tf.contrib.layers.l2_regularizer(scale=self.weight_decay)
        return tf.get_variable(name=name, shape=shape,
                               initializer=tf.contrib.layers.xavier_initializer(seed=19))


if __name__ == '__main__':
    import os
    pnet = ProjectionNet({"verbose":True, "input_shape":[None,95,69,79]})
    pnet.build_network(tf.Session())
    g = tf.summary.merge_all()
    f_wtr = tf.summary.FileWriter(os.getcwd()+"/summary_file")
    with tf.Session() as sess:
        f_wtr.add_graph(sess.graph)
    input("wait")
