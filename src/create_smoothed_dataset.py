#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 21:06:52 2018

@author: abhijit
"""

#for Creating smoothed dataset

import data
import numpy as np
import tensorflow as tf
import os
import h5py as hf

def get_3d_gaussian_kernel (size=[4,4,4], sigma=1.0, pixel_unit=1):
    ax_range = []
    for i in range(3):
        if size[i]%2==0:
            high = size[i]/2
            low = -high+1
            ax_range.append(np.arange(low,high+1,pixel_unit))
    xx, yy, zz = np.meshgrid(ax_range[0], ax_range[1], ax_range[2])
    kernel = np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))
    kernel = kernel/np.sum(kernel)
    kernel = np.expand_dims(kernel,axis=-1)
    kernel = np.expand_dims(kernel,axis=-1)
    return kernel


if False:
    data_range = [[0,256]]
    #data_range = [[0,1]]
    X , Y = data.get_parkinson_classification_data(ranges=data_range)
    _ , M = data.get_parkinson_TF_data (ranges=data_range)
    
    
    
    window = get_3d_gaussian_kernel()
    
    gpu = 1
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    
    graph = tf.Graph()
    
    with graph.as_default():
        W_1 = tf.constant(value=window, dtype=tf.float32)
        x_in = tf.placeholder(shape=[None,95,69,79,1],dtype=tf.float32)
        strides = [1, 1, 1, 1, 1]
        smooth = tf.nn.conv3d(x_in, W_1, strides, "SAME", name="conv_1")
    
    config = None
    with tf.Session(graph=graph, config=config) as sess:
        with hf.File("smoothed_parkinson_with_tf_data.h5", "w") as f:
            f.create_dataset("volumes",shape=(257,95,69,79,1),dtype=np.float32)
            f.create_dataset("tf_maps",shape=(257,95,69,1),dtype=np.float32)
            #f.create_dataset("labels",shape=(257,1),dtype=np.float32)
            f.create_dataset("one_hot_labels",shape=(257,3,1),dtype=np.float32)
            for i in range(X.shape[0]):
                smooth_vol = sess.run(smooth, feed_dict={x_in:X[i:i+1,:,:,:,:]})
                f["volumes"][i] = smooth_vol[0]
                f["tf_maps"][i] = M[i]
                f["one_hot_labels"][i] = Y[i]
                print("Sample ",i)
                
        

X , M = data.get_pretraining_TF_data()



window = get_3d_gaussian_kernel()

gpu = 1
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

graph = tf.Graph()

with graph.as_default():
    W_1 = tf.constant(value=window, dtype=tf.float32)
    x_in = tf.placeholder(shape=[None,95,69,79,1],dtype=tf.float32)
    strides = [1, 1, 1, 1, 1]
    smooth = tf.nn.conv3d(x_in, W_1, strides, "SAME", name="conv_1")

config = None
with tf.Session(graph=graph, config=config) as sess:
    with hf.File("smoothed_pretraining_tensor_factorized_data.h5", "w") as f:
        f.create_dataset("volumes",shape=(1077,95,69,79,1),dtype=np.float32)
        f.create_dataset("tf_maps",shape=(1077,95,69,1),dtype=np.float32)
        #f.create_dataset("labels",shape=(257,1),dtype=np.float32)
        #f.create_dataset("one_hot_labels",shape=(257,3,1),dtype=np.float32)
        for i in range(X.shape[0]):
            smooth_vol = sess.run(smooth, feed_dict={x_in:X[i:i+1,:,:,:,:]})
            f["volumes"][i] = smooth_vol[0]
            f["tf_maps"][i] = M[i]
            #f["one_hot_labels"][i] = Y[i]
            print("Sample ",i)
            

























