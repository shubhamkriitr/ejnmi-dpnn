#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 22:11:52 2018
Running PXNet-Fisrt Half for pretraining of PXNet-Full
@author: shubham
"""
import tensorflow as tf
import logger
import data
import net
import numpy as np
import utility as ut
import os
from PXNET_GAP import ProjectionNet
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
Y_whole specifications:
<class 'numpy.ndarray'>
(1334, 95, 69, 1)
float32
Mean 4.994355
Median 5.865527
Max 11.497833
Min -0.13825107
std.dev. 2.7348104

"""
mn_v = -3173.77#min value
mx_v = 95617.8#max value


max_fold = 5

data_range_pd = [(0,90), (91,120), (121,256)]#parts of the parkinsond dataset to be used
initial_set = 1
lrs = [5*1e-5,1e-5,1e-4,0.25*1e-4,0.5*1e-4]

set_num = initial_set-1
for lr in lrs:
    set_num+=1
    for fold in range(1,max_fold+1):
        #%%DATA
        #fetch data
        sp_pd = ut.get_split_ranges(data_range_pd,fold,max_fold)
        r_pd = sp_pd["train"]
        s_pd = sp_pd["val"]
        
        X_pd , Y_pd = data.get_parkinson_classification_data(ranges=r_pd)
        X_val_pd, Y_val_pd = data.get_parkinson_classification_data(ranges=s_pd)
        
        X_pd = X_pd[:,:,:,:,0]
        X_val_pd = X_val_pd[:,:,:,:,0]
        
        X = X_pd
        Y = Y_pd[:,:,0]
        X_val = X_val_pd
        Y_val = Y_val_pd[:,:,0]
        
        print ("Val Input Shape",X_val.shape)
        log_list.append({"Val Input Shape":X_val.shape})
        log_list.append({"training on":r_pd, 'validating_on':s_pd})
        log_list.append({"X":X, "Y":Y, "X_val":X_val, "Y_val":Y_val})
        
        
        #mormalize
        X = (X-mn_v)/(mx_v-mn_v)
        X_val = (X_val-mn_v)/(mx_v-mn_v)
        train_gen = data.DataGenerator(X,Y)
        val_gen = data.DataGenerator(X_val,Y_val)

        
        del X
        del Y
        del X_val
        del Y_val








