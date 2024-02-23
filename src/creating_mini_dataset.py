#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 22:46:53 2017

@author: shubham
"""

import data
import tensorflow as tf
import numpy as np
import h5py as hf

def get_avg_pooling_layer (in_layer,pool_size):
    ksz = [pool_size, pool_size, pool_size]
    ans = tf.layers.average_pooling3d(in_layer,ksz,ksz,padding='valid')
    return ans

X_in, Y_in = data.get_data(ranges=[[0,0]])
print (X_in.shape,Y_in.shape)

X = tf.placeholder(dtype=tf.float32,shape=[None,256,256,256,1],name="Input")
pooled = get_avg_pooling_layer(X,8)
pooled4 = get_avg_pooling_layer(X,4)
with tf.Session() as sess:
    with hf.File("train_small.h5",'w') as f:
        f.create_dataset( "data",shape=(500,32,32,32,1),dtype=np.float32)
        for i in range(500):
            print("Train",i)
            X_in, Y_in = data.get_data(ranges=[[i,i]])
            Z = sess.run(pooled,{X:X_in})
            f['data'][i:i+1] = Z

    with hf.File("val_small.h5",'w') as f:
        f.create_dataset( "data",shape=(88,32,32,32,1),dtype=np.float32)
        for i in range(500,588):
            print("Val",i)
            X_in, Y_in = data.get_data(ranges=[[i,i]])
            Z = sess.run(pooled,{X:X_in})
            f['data'][i:i+1] = Z


with tf.Session() as sess:
    with hf.File("train_64.h5",'w') as f:
        f.create_dataset( "data",shape=(500,64,64,64,1),dtype=np.float32)
        for i in range(500):
            print("Train",i)
            X_in, Y_in = data.get_data(ranges=[[i,i]])
            Z = sess.run(pooled4,{X:X_in})
            f['data'][i:i+1] = Z

    with hf.File("val_64.h5",'w') as f:
        f.create_dataset( "data",shape=(88,64,64,64,1),dtype=np.float32)
        for i in range(500,588):
            print("Val",i)
            X_in, Y_in = data.get_data(ranges=[[i,i]])
            Z = sess.run(pooled4,{X:X_in})
            f['data'][i:i+1] = Z

sess = tf.Session()
Z = sess.run(pooled,{X:X_in})
print(Z.shape,type(Z), Z.dtype)
Z = sess.run(pooled4,{X:X_in})
print(Z.shape,type(Z), Z.dtype)