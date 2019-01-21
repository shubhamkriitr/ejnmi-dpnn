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

def XENT  (y_true, y_pred, scope="CENT"):
    with tf.variable_scope(scope):
        y_pred = tf.clip_by_value(y_pred,1e-7,1.0)
        cost = - tf.reduce_mean(y_true*tf.log(y_pred) +
                                (1-y_true)*tf.log(1-y_pred+1e-7))
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


def SSIM_UK(img1, img2, cs_map=False, mean_metric=True, size=6,dynamic_range=1):
    with tf.variable_scope("SSIM_UK"):
        window = tf.ones([size,size,1,1],dtype=tf.float32,name="UK")
        window = window/tf.reduce_sum(window)
        K1 = 0.01
        K2 = 0.03
        L = dynamic_range  # depth of image (255 in case the image has a differnt scale)
        C1 = (K1*L)**2
        C2 = (K2*L)**2
        mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
        mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
        mu1_sq = mu1*mu1
        mu2_sq = mu2*mu2
        mu1_mu2 = mu1*mu2
        sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
        sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
        sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
        if cs_map:
            value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                        (sigma1_sq + sigma2_sq + C2)),
                    (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
        else:
            value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                        (sigma1_sq + sigma2_sq + C2))

        if mean_metric:
            value = tf.reduce_mean(value)
    return value


def dice_loss (y_true, y_pred):
    pass


if __name__ == '__main__':
    x = tf.placeholder(dtype=tf.float32)
    y = tf.constant(3,dtype=tf.float32)
    mae = MAE(x,y,"COST")
    sess = tf.Session()
    print(sess.run(mae,feed_dict={x:1.0}) )
    print(sess.run(mae,feed_dict={x:2.0}) )
    print(sess.run(mae,feed_dict={x:3.0}) )