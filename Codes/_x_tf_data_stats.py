#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 08:20:57 2018

@author: reuter
"""

#STATS

import tensorflow as tf
import logger
import data
import net
import numpy as np
import utility as ut
import os
from PretrainingPXNetFirstHalf import ProjectionNetFirstHalf
import matplotlib.pyplot as plt


"""
train specifications:
<class 'numpy.ndarray'>
(1077, 95, 69, 79)
float32
Mean 10282.7
Median 7646.11
Max 95617.8
Min -3173.77
std.dev. 9765.27
----------
----------
Y_train specifications:
<class 'numpy.ndarray'>
(1077, 95, 69, 1)
float32
Mean 4.99517
Median 5.86557
Max 11.3836
Min -0.138251
std.dev. 2.73669


Y_whole specifications:
<class 'numpy.ndarray'>
(1334, 95, 69, 1)
float32
Mean 4.994355
Median 5.865527
Max 11.497833
Min -0.13825107
std.dev. 2.7348104
----------
"""
X , Y = data.get_pretraining_TF_data(ranges=[[0,1076]])
print ("Input Shape",X.shape)
X = X[:,:,:,:,0]
def get_freq_dist (X, scale=None,fig_file_name="Frequency Distribution.png"):
    if scale is not None:
        xn = (X/scale).astype(np.int16)
    else:
        xn = X
    
    v,fq = np.unique(xn, return_counts=True)
    
    fig,ax = plt.subplots(1,1)
    ax.set_title("Frequency Distribution")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Values in {}".format(scale))
    ax.plot(v,fq)
    plt.show()
    fig.savefig (fig_file_name)

X_pd , Y_pd = data.get_parkinson_TF_data()
#ut.get_array_info(Y_pd,"Y_pd")
#get_freq_dist(Y_pd,1.0,"TF_PARKINSON.png")

Y = np.concatenate([Y,Y_pd],axis=0)
ut.get_array_info(Y,"Y_whole")
get_freq_dist(Y_pd,1.0,"TF_DATA_ALL_1077+257.png")

tf_mx = 11.497833
tf_mn = -0.13825107
Y = (Y-tf_mn)/(tf_mx-tf_mn)
Y = np.clip(Y,1e-7,1.0-1e-7)
ut.get_array_info(Y,"Y_whole_normalized")
get_freq_dist(Y,0.1,"TF_DATA_ALL_1077+257_normalized.png")





























