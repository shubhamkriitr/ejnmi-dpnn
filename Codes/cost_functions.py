#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 17:03:39 2017

@author: shubham
"""

import tensorflow as tf

def MAE (y_true, y_pred, scope = "MAE"):
    """Defines a node representing Mean Absloute Error b/w y_true and y_pred
        tensors.

        Args:
            y_true: label tensor
            y_pred: prediction tensor
            scope: name scope

        Returns:
            A node representing Mean Absloute Error b/w y_true and y_pred
            tensors.
    """
    with tf.variable_scope(scope):
        cost = tf.reduce_mean(tf.losses.absolute_difference(y_true, y_pred))
    return cost

def MSE (y_true, y_pred, scope="MSE"):
    with tf.variable_scope(scope):
        cost = tf.losses.mean_squared_error(y_true, y_pred)
    return cost

# def XENT  (y_true, y_pred, scope="CENT"):
#     with tf.variable_scope(scope):
#         y_pred = tf.clip_by_value(y_pred,1e-7,1.0)
#         cost = - tf.reduce_mean(y_true*tf.log(y_pred))
#     return cost

def INV_WEIGHTED_XENT  (y_true, y_pred, scope="INV_WTD_XENT"):
    """USing 1-n_i/N as weight for ith class"""
    wts = [ 0.66260159,  0.88211381,  0.45528454]
    print("USING INV_WEIGHT : ", wts)
    with tf.variable_scope(scope):
        wt_tnsr = tf.constant(wts, dtype=tf.float32)
        y_pred = tf.clip_by_value(y_pred,1e-7,1.0)
        wtdy_true = wt_tnsr*y_true
        cost = - tf.reduce_mean(wtdy_true*tf.log(y_pred))
    return cost

def WTD_XENT  (y_true, y_pred, scope="CENT",WTS=[0.00019290123456790122, 0.001736111111111111, 8.573388203017832e-05]):
    with tf.variable_scope(scope):
        y_pred = tf.clip_by_value(y_pred,1e-7,1.0)
        cost = -1.0*WTS[0]*tf.reduce_mean(y_true[:,0]*tf.log(y_pred[:,0]) +
                                (1-y_true[:,0])*tf.log(1-y_pred[:,0]+1e-7))
        for i in range(1,len(WTS)):
            cost = cost - WTS[i]*tf.reduce_mean(y_true[:,i]*tf.log(y_pred[:,i]) +
                                    (1-y_true[:,i])*tf.log(1-y_pred[:,i]+1e-7))

    return cost


def dice_loss (y_true, y_pred):
    pass


if __name__ == '__main__':
    # x = tf.placeholder(dtype=tf.float32)
    # y = tf.constant(3,dtype=tf.float32)
    # mae = MAE(x,y,"COST")
    # sess = tf.Session()
    # print(sess.run(mae,feed_dict={x:1.0}) )
    # print(sess.run(mae,feed_dict={x:2.0}) )
    # print(sess.run(mae,feed_dict={x:3.0}) )
    import numpy as np
    yt = tf.placeholder(dtype=tf.float32, shape=(None, 3))
    yp = tf.placeholder(dtype=tf.float32, shape=(None, 3))
    cost = INV_WEIGHTED_XENT(yt, yp, scope="INV_WEIGHTED_XENT")
    ypi = [[1,0,0],[1,1,1],[1,1,0]]
    yti = [[1,0,1],[1,1,1],[1,1,0]]
    ypi = np.array(ypi, dtype=np.float32)
    yti = np.array(yti, dtype=np.float32)
    with tf.Session() as sess:
        c = sess.run([cost], feed_dict={yt:yti,yp:ypi})
        print(c)
