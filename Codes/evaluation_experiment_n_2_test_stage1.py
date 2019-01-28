#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 12:28:48 2018

@author: abhijit
"""

#EVALUATE MODELS OF EXPERIMENT 15
import tensorflow as tf
import logger
import data
import net
import numpy as np
import utility as ut
import os
import test_performance_calculator as pc
from copy import deepcopy
import fold_wise_bar_plot as fwbp
import time

#%%Select the network
from PXNET_SIGMOID_GAP import ProjectionNet as network

#%%Fetch data
max_fold = 5


print("-----*****Change Min-Max values*****-----")
mn_v = 0.0#min value
mx_v = 92152.0#max value

# Max 41896.2 # Actual min & max of test dataset
# Min 0.0

# data_ranges=[(0,62),] Using default
X ,  = data.get_new_test_parkinson_cls_data(ranges=[[0, 62]])

X = X[:,:,:,:,0]
print("-----*****Change Min-Max values*****-----")
X = (X-mn_v)/(mx_v-mn_v)

model_dir = os.getcwd()+os.sep+"Checkpoints"

def create_test_score_files(model_dir,name="Model"):
    op_dir = model_dir+"_Out"
    if not os.path.exists(op_dir):
        os.mkdir(op_dir)
    pc.calculate_and_store_test_scores(m_dir,op_dir,name,network,{},X,suffix="_testscore")



#Make changes here
selected = []


for slctd in selected:
    name = "PXNET_SIGMOID_GAP"
    root_model_dir = os.getcwd()+os.sep+"Checkpoints"+os.sep+slctd    
    L = os.listdir(root_model_dir)
    for folder in L:
        m_dir = root_model_dir+os.sep+folder
        create_test_score_files(m_dir,name)# Works fine only for five folded evaluation. -- #TODO need changes
















