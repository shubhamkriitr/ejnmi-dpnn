#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 23:28:19 2018

@author: shubham
"""

import tensorflow as tf
import logger
import data
import net
import numpy as np

with tf.variable_scope("Holder"):
    x = tf.placeholder(dtype=tf.float32,name = "X_input")
    y = [x]
    z = y.copy()
    print(y is z)
    print(y[0] is z[0])

lr = 0.001
epochs = 20
save_step = 5


class MiniNet (net.Net):
    def __init__(self):
        super().__init__({})
        self.scope = "MiniNet"
        self.node_dict = {}
    def _build_network(self):
        with tf.variable_scope(self.scope):
            x = tf.placeholder(dtype=tf.float32,shape=(None,1),name="X_in")
            y_true = tf.placeholder(dtype=tf.float32,shape=(None,1),name="Y_true")
            w = tf.get_variable(name="W",initializer=tf.constant(0,dtype=tf.float32))
            w2 = tf.get_variable(name="W2",initializer=tf.constant(0.1,dtype=tf.float32))
            self.w3 = tf.get_variable(name="W3",initializer=tf.constant([0,10,0],dtype=tf.float32))
            self.update_w = self.w3.assign_add(tf.constant([0.1,-0.1,0.2],dtype=tf.float32))
            self.w4 = tf.get_variable(name="W4",initializer=tf.constant([[0]],dtype=tf.float32))
            self.update_w4 = self.w4.assign_add(tf.constant([[1]],dtype=tf.float32))
            hidden = tf.multiply(x,w,name="mult")
            y_pred = tf.multiply(hidden,w2,name="mult2")
            self.cost = tf.reduce_mean(tf.squared_difference(y_true,y_pred))
            self.inputs.append(x)
            self.outputs.append(y_pred)
            self.true_outputs.append(y_true)
            self.histogram_list = [tf.summary.histogram("w1",w),
                                   tf.summary.histogram("w2",w2),
                                   tf.summary.histogram("w3",self.w3)]
            self.node_dict["w"] = w
            self.node_dict["w2"] = w2
            self.node_dict["w4_into_x"] = tf.reduce_mean(x*self.w4)
            self.node_dict["MSE"] = self.cost

model = MiniNet()
model.build_network()
#fwtr = tf.summary.FileWriter("summary_x_testing_name_scopes_in_tf")
#fwtr.add_graph(tf.get_default_graph())
with model.graph.as_default():
    lr = tf.placeholder(dtype=tf.float32,shape=[])
    opti = tf.train.GradientDescentOptimizer(lr)
    train_op = opti.minimize(model.cost)

def get_lr (epoch):
    return 0.001-0.00001*(epoch-1)

N = 20
X = np.array(list(range(0,N)),dtype=np.float32).reshape(N,1)
Y = np.array(list(range(0,N)),dtype=np.float32).reshape(N,1)

train_gen = data.DataGenerator(X,Y,True)
val_gen = data.DataGenerator(X[(int)(N/2):N],Y[(int)(N/2):N],True)

logger_args =  {"scope": "LOGS",
                "model": model ,
                "session":None ,
                "log_root_folder":"LOGS",
                "bsz":5,
                "train_gen":train_gen,
                "val_gen":val_gen}
lgr = logger.Logger(logger_args)


with model.graph.as_default():
    lgr.create_summary_ops(model.node_dict, {"LR":lr},model.histogram_list)

with tf.Session(graph = model.graph) as sess:
    lgr.set_session(sess)
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        print(i+1,"W",sess.run([model.node_dict["w"],model.node_dict["w2"],
                                model.w3, model.w4]))
        fd = {model.inputs[0]:X,model.true_outputs[0]:Y,lr:get_lr(i+1)}
        lgr.record_train_summary(i+1,{lr:fd[lr]})
        lgr.record_val_summary(i+1,{lr:fd[lr]})
        print("cost",sess.run([model.cost,model.update_w, model.update_w4, train_op],fd))

