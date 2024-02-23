# !/usr/bin/env python3
#  -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 05:21:46 2017

@author: shubham
"""

from tensorflow.python.layers.core import Flatten, dense
import tensorflow as tf
import os
import time

NF = 8  # The number of filters in the first block
# doubles after every max-pooling operation (In total 5 such blocks before FC layer)

def _dbmsg (msg="Message", obj = None):
    """for debugging"""
    if 't_base' not in _dbmsg.__dict__:
        _dbmsg.t_base = time.time()
    print("At t =",time.time()-_dbmsg.t_base,":",msg,end=" ")
    if obj is not None:
        print(obj)


class AgeRegNet:

    def __init__(self,input_shape,output_shape, dtype=tf.float32, model_path=None,
                 old_model_path=None, scope_name="AgeRegNet",
                 load_pretrained = False):
        # add_modification_options
        self.scope = scope_name
        self.num_of_inputs = 1
        self.data_format = "channels_last"
        self.ksz = 3  # kernel size
        self.dtype = tf.float32
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.load_pretrained = load_pretrained# flag representing whether or not to load
                                    # the saved model(if exists)
        self.old_model_path = old_model_path
        self.trainable = True   # can be overridden if it is true
        self.x_1 = None
        self.y_true_1 = None
        self.y_pred_1 = None
        self.cost_1 = None
        self.saver = None
        self.inputs = None
        self.outputs = None
        self.costs = None
        self.labels = None
        self.built_flag = False

    def build_network(self,sess):
        """Should be explicitly called in a session to inialize the graph weig-
        hts or load it from the disk"""
        _dbmsg("Starting Build Operation")
        with tf.variable_scope(self.scope):
            self.x_1 = tf.placeholder(self.dtype,shape=self.input_shape,name="x_1")
            self.y_true_1 = tf.placeholder(self.dtype,shape=self.output_shape,name="y_true_1")
            conv_block_1 = self.conv_block(self.x_1, [NF, NF],1)
            conv_block_2 = self.conv_block(conv_block_1, [NF*2, NF*2],2)
            conv_block_3 = self.conv_block(conv_block_2, [NF*4, NF*4],3)
            conv_block_4 = self.conv_block(conv_block_3, [NF*8, NF*8],4)
            conv_block_5 = self.conv_block(conv_block_4, [NF*16, NF*16],5)

            _dbmsg("conv_blocks_built")
            self.y_pred_1 = self.fc_block(conv_block_5, 1)
            self.cost_1 = tf.reduce_mean(tf.losses.absolute_difference (self.y_true_1,
                                                               self.y_pred_1))
            self.saver = tf.train.Saver()
        _dbmsg("Network Created/about to load weights if exists")
        with sess.as_default():
                sess.run(tf.global_variables_initializer())
                self.built_flag = True
        self.inputs = [self.x_1]
        self.outputs = [self.y_pred_1]
        self.labels = [self.y_true_1]
        self.costs = [self.cost_1]

    def load_network (self,sess):
        if (self.load_pretrained):
            if not self.built_flag:
                self.build_network(sess)
            with sess.as_default():
                print("Checkpoint at:",self.old_model_path,"loaded.")
                self.saver.restore(tf.get_default_session(), self.old_model_path)
        else:
            print("loading network not allowed")


    def conv_block(self, in_layer, num_out_filters, name):
        scope = "conv_block_{}".format(name)
        with tf.variable_scope(scope):
            conv_block = self._conv3_relu_conv3_bn_relu_mp(scope, in_layer,
                                                           num_out_filters)
        return conv_block


    def fc_block(self, in_layer, name):

        # ====================================================================
        #         in_size = 1
        #         shape_tensor = in_layer.get_shape()
        #         assert(len(shape_tensor)>=2)
        #         for in in range (1,len(shape_tensor)):
        #             in_size = in_size * shape_tensor[i].value
        # ====================================================================
        with tf.variable_scope("fc_block_{}".format(name)):
            flattened = Flatten(name="flatten_1")(in_layer)
            out = dense(flattened, 1, activation=None)# Linear Activation
        return out

    def _conv3_relu_conv3_bn_relu_mp(self, scope, in_layer, num_filters=[64,
                                                                         64]):
        """Main conv block"""
        channels = in_layer.get_shape()[-1].value
        with tf.variable_scope(scope):
            shape = [self.ksz, self.ksz, self.ksz, channels,
                     num_filters[0]]
            strides = [1, 1, 1, 1, 1]
            W_1 = self._get_weight_and_bias_vars("weight_1", shape)
            tf.summary.histogram("W1",W_1)
            conv_1 = tf.nn.conv3d(in_layer, W_1, strides, "SAME", name="conv_1")
            # conv3d data_format is NDHWC(make sure to pass input accordingly)
            bias_1 = self._get_weight_and_bias_vars("bias_1", [shape[4]])
            tf.summary.histogram("b1",bias_1)
            #print ("bias_1",bias_1.get_shape().as_list())
            layer_1 = tf.nn.bias_add(conv_1, bias_1)
            relu_1 = tf.nn.relu(layer_1, name="relu_1")

            shape = [self.ksz, self.ksz, self.ksz, num_filters[0],
                     num_filters[1]]
            W_2 = self._get_weight_and_bias_vars("weight_2", shape)
            tf.summary.histogram("W2",W_2)
            conv_2 = tf.nn.conv3d(relu_1, W_2, strides, "SAME", name="conv_2")
            bias_2 = self._get_weight_and_bias_vars("bias_2", [shape[4]])
            tf.summary.histogram("b2",bias_2)
            layer_2 = tf.nn.bias_add(conv_2, bias_2)
            batch_norm_1 = tf.contrib.layers.batch_norm(layer_2)
            # TODO_Replace contrib functions with stable versions
            # bn expects NHWC format-- NDHWC will work, since C is at the end,
            # in both of them
            relu_2 = tf.nn.relu(batch_norm_1, name="relu_1")
            max_pool_1 = tf.layers.max_pooling3d(relu_2,[2,2,2],[2,2,2],
                                                         padding="SAME",name="max_pool_1")
# =============================================================================
#             max_pool_1 = tf.layers.pooling.max_pooling3d(relu_2,[2,2,2],[2,2,2],
#                                                          padding="SAME",name="max_pool_1")
# =============================================================================
        return max_pool_1

    def _get_weight_and_bias_vars(self, name, shape):
        """ shape must be of the form: [filter_depth, filter_height,
        filter_width, in_channels, out_channels], uses scope of the calling
        function. """
        return tf.get_variable( name=name, shape=shape,
                        initializer=tf.contrib.layers.xavier_initializer() )
        #TODO_ Replce contrib functions with stable versions



if __name__ == '__main__':
    a = AgeRegNet([None,50,50,50,1],[None,1])
    sess1 = tf.Session()
    print ("session_name at start:",sess1,type(sess1))
    a.build_network(sess1)
