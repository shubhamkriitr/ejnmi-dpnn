# !/usr/bin/env python3
#  -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 15:21:46 2017

@author: shubham
"""
import tensorflow as tf
import utility as ut
from tensorflow.python.layers.core import Flatten, dense
from net import Net
from cost_functions import MAE, MSE, XENT, WTD_XENT

cost_functions = {"MAE":MAE, "MSE":MSE, "XENT":XENT, "WTD_XENT":WTD_XENT}

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
        self.scope = "ProjectionNet"
        self.cost_function = "XENT"
        self.pretraining_cost_function = "MSE"
        self.weight_decay = None
        self.print("="*25)
        self.input_shape = None
        self.output_shape = [None,3]
        self.pretraining_output_shape = [None, 95, 69, 1]
        self._init_all(arg_dict)
        self.labels = self.true_outputs# for compatibility with other old scripts
        self.log_list = self.histogram_list# for compatibility with other old scripts
        self.pt_vars = []#List of pretraining variables
        self.ft_vars = []#List of vars to be fine tuned i.e. part of the network
        #after compression part

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

            self.pt_vars = tf.trainable_variables()
            print ("\n\nTrainable variables:\n")
            ut.print_sequence(self.pt_vars)
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
            block_10 = tf.layers.average_pooling2d(block_9,[4,4],[4,4])
            print ("block_10:",block_10.get_shape())
            block_11 = tf.layers.conv2d(block_10,3,[1,1])
            self.print("block_11",block_11.get_shape())
            flattened = tf.reshape(block_11,[-1,3]) #Flatten(name="flatten_1")(block_12)
            self.print("flattened:",flattened.get_shape())

            all_trainable_vars = tf.trainable_variables()
            ut.get_list_difference(all_trainable_vars,self.pt_vars,self.ft_vars)
            print ("\n\nOther Trainable variables:\n")
            ut.print_sequence(self.ft_vars)
            y_pred = tf.nn.softmax(flattened,name="softmax")
            self.print("y_pred:",y_pred.get_shape())
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
        assert((len(self.pt_vars)+len(self.ft_vars))==len(tf.trainable_variables()))

    def build_pretraining_network(self, sess):
        """Method to create the compression part of the graph for pretraining."""
        with tf.variable_scope(self.scope):
            input_layer = tf.placeholder(dtype=tf.float32, shape = self.input_shape)
            y_true = tf.placeholder(dtype=tf.float32, shape=self.pretraining_output_shape)
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
            y_pred = block_5
            self.pt_vars = tf.trainable_variables()
            print ("\n\nTrainable variables:\n")
            ut.print_sequence(self.pt_vars)
            cost = cost_functions[self.cost_function](y_true, y_pred)
            self.outputs.append(y_pred)
            self.print("CALCULATED OUTPUT SHAPE:",y_pred.get_shape())
            self.costs.append(cost)
            self.print("Pretraining Network Built")
        with sess.as_default():
            sess.run(tf.global_variables_initializer())
            self._built_flag = True
        self.saver = tf.train.Saver()
        assert(len(self.pt_vars)==len(tf.trainable_variables()))

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
    sess = tf.Session()
    pnet.build_network(sess)
#    pnet.build_pretraining_network(sess)
    g = tf.summary.merge_all()
    f_wtr = tf.summary.FileWriter(os.getcwd()+"/summary_file")
    f_wtr.add_graph(sess.graph)
    sess.close()
    input("wait")
