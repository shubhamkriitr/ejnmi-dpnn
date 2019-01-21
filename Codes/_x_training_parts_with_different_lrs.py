#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 22:16:16 2018

@author: shubham
"""

import tensorflow as tf
import numpy as np
x_in = [[1, 1, 1],
        [2, 2, 2],
        [1, 2, 2],
        [3, 4, 5]]
x_in = np.array(x_in, dtype=np.float32)

y_t = [[4,4,4],[8,8,8],[6,6,6],[15,15,15]]
y_t = np.array(y_t, dtype=np.float32)

W1_init = tf.constant([[2,1,1]],
                      dtype=tf.float32, shape = (3,1))
#tf.ones(shape=(3,1), dtype=tf.float32)

x = tf.placeholder(shape=[None,3], dtype=tf.float32)
y_true = tf.placeholder(shape=[None,3], dtype=tf.float32)
print(tf.trainable_variables())

W = tf.get_variable(name="W1", dtype=tf.float32, initializer=W1_init)
print("After W:", tf.trainable_variables())
var_list_1 = (tf.trainable_variables())

W2 = tf.get_variable(name="W2",
                     dtype=tf.float32,
                     initializer=0.5*tf.ones(shape=(1,3),
                     dtype=tf.float32))
print("After W2:", tf.trainable_variables())


W3 = tf.get_variable( name="W3", dtype=tf.float32,
                     initializer=tf.eye(3,dtype=tf.float32),
                     trainable=True)

var_list_2 = (tf.trainable_variables())
l = []

for var in var_list_2:
    if var in var_list_1:
        print("var:",var,"in list 1")
    else:
        print("var:",var,"is only in list 2")
        l.append(var)

del var_list_2
var_list_2 = l

h1 = tf.matmul(x,W)
h2 = tf.matmul(h1,W2)
y_pred = tf.matmul(h2,W3)
err = tf.reduce_mean(tf.squared_difference(y_true,y_pred))
print("After err:", tf.trainable_variables())

optimizer1 = tf.train.GradientDescentOptimizer(0.00)
optimizer2 = tf.train.GradientDescentOptimizer(0.001)

grads = tf.gradients(err, var_list_1 + var_list_2)
grads1 = grads[:len(var_list_1)]
grads2 = grads[len(var_list_1):]
print("After optimizer:", tf.trainable_variables())

train_op1 = optimizer1.apply_gradients(zip(grads1,var_list_1))
train_op2 = optimizer2.apply_gradients(zip(grads2,var_list_2))
train_step = tf.group(train_op1, train_op2)

epochs = 1
with tf.Session() as  sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epochs):
        print("Epoch:",i+1,"--"*15,
              "\nW:\n",sess.run(W),"\nW2:\n",sess.run(W2),
              "\n","\nW3:\n",sess.run(W3),"\n")
        ans = sess.run([train_step,y_pred,grads1,grads2,err],
                       feed_dict={x:x_in,y_true:y_t})
        names = ["train_step", "y_pred","grads1" ,"grads2" ,"err"]
        for i in range(len(names)):
            print("\n\n"+names[i]+"\n",ans[i])


print ("Finally","\nL1:\n\n",var_list_1,"\nL2:\n\n",var_list_2)













