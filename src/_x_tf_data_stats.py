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
# X , Y = data.get_pretraining_TF_data(ranges=[[0,1076]])
# print ("Input Shape",X.shape)
# X = X[:,:,:,:,0]
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

# X_pd , Y_pd = data.get_parkinson_TF_data()
# #ut.get_array_info(Y_pd,"Y_pd")
# #get_freq_dist(Y_pd,1.0,"TF_PARKINSON.png")

# Y = np.concatenate([Y,Y_pd],axis=0)
# ut.get_array_info(Y,"Y_whole")
# get_freq_dist(Y_pd,1.0,"TF_DATA_ALL_1077+257.png")

# tf_mx = 11.497833
# tf_mn = -0.13825107
# Y = (Y-tf_mn)/(tf_mx-tf_mn)
# Y = np.clip(Y,1e-7,1.0-1e-7)
# ut.get_array_info(Y,"Y_whole_normalized")
# get_freq_dist(Y,0.1,"TF_DATA_ALL_1077+257_normalized.png")
def test_1():
    def plot_hist(arr, ax, title):
        ax.set_title(title)
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        ax.hist(arr.flatten(), bins='auto')

    ##Figure
    fig, axes = plt.subplots(5, 1)
    ####### New dataset
    ## DEV
    X_new, Y_new = data.get_new_dev_parkinson_cls_data()
    X_new = X_new[:,:,:,:,0]
    print(X_new.shape)

    mn_v = 0.0#min value
    mx_v = 92152.0#max value

    X_new_norm = (X_new - mn_v)/(mx_v-mn_v)
    mx_v = 41896.2
    X_new_norm_test = (X_new- mn_v)/(mx_v-mn_v)
    plot_hist(X_new, axes[0], 'X_new')
    plot_hist(X_new_norm, axes[1], 'X_new_norm')
    plot_hist(X_new_norm_test, axes[2], 'X_new_norm_t')

    X_new_test, = data.get_new_test_parkinson_cls_data()

    mn_v = 0.0#min value
    mx_v = 92152.0#max value
    X_new_test_norm_dev = (X_new_test- mn_v)/(mx_v-mn_v)

    mx_v = 41896.2
    X_new_test_norm_test = (X_new_test- mn_v)/(mx_v-mn_v)


    plot_hist(X_new_test, axes[3], 'X_new_test')
    plot_hist(X_new_test_norm_dev, axes[4], 'X_new_test_norm_d')
    plot_hist(X_new_test_norm_test, axes[5], 'X_new_test_norm_t')


    plt.show()

def calculate_weights(Y):
    Y = Y[:,:,0]
    sm = np.sum(Y, axis=0)
    wts = (1-sm/np.sum(sm))
    print(sm)
    print(wts)

if __name__ == '__main__':
    X_new, Y_new = data.get_new_dev_parkinson_cls_data()
    # ut.get_array_info(Y_new)
    calculate_weights(Y_new)
    





























