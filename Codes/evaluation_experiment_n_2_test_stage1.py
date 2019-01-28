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
import performance_calculator_bin_search as pc
from copy import deepcopy
import fold_wise_bar_plot as fwbp
import time

#%%Select the network
from PXNET_SIGMOID_GAP import ProjectionNet as network

name = "PXNET_SIGMOID_GAP"
#%%Fetch data
break_pts = [(0,82), (83,111), (112,245)]
max_fold = 5


print("-----*****Change Min-Max values*****-----")
mn_v = 0.0#min value
mx_v = 92152.0#max value

# data_ranges=[(0,62),] Using default
X ,  = data.get_new_test_parkinson_cls_data(ranges=[[0, 62]])

X = X[:,:,:,:,0]
print("-----*****Change Min-Max values*****-----")
X = (X-mn_v)/(mx_v-mn_v)

#%%For sorting out best models

model_dir = os.getcwd()+os.sep+"Checkpoints"

eps 
fl = [{"SET_?":{
                        1:["epoch_?"],# best ones
                        2:["epoch_?"],
                        3:["epoch_?"],
                        4:["epoch_?"],
                        5:["epoch_?"],
                        
            }},]
file_keys_list = fl

response = input("Select models?:(type `no` and `enter` to skip)"+str(file_keys_list))
folder_list = []
if response.lower() == "yes":
    for f_keys in file_keys_list:
        print("-"*12)
        print(str(f_keys))
        print("-"*12)
        key = list(f_keys.keys())[0]
        tag =  key + "_" + f_keys[key][1][0]
        fldr = pc.pick_models(model_dir,f_keys,max_fold,search_level=3,tag=tag,tagsep="_tgxtest_")
        folder_list.append(fldr)
        print("x"*25)
        time.sleep(1.1)# wait for ut.apppend_time_string() - to avoid same folder name
# wait = input("wait-- press enter to proceed or Ctrl + C to abort.")

#%% Calculate Scores


def create_score_files_and_plots(model_dir,name="Model"):
    op_dir = model_dir+"_Out"
    if not os.path.exists(op_dir):
        os.mkdir(op_dir)
    
    pc.calculate_and_store_scores(m_dir,op_dir,name,network,{},X,Y,break_pts)
    ip_dir  = op_dir
    op_dir = ip_dir+os.sep+"scores"
    if not os.path.exists(op_dir):
        os.mkdir(op_dir)
    pc.calculate_and_store_cf_metrics(ip_dir,op_dir,"score",has_image=False)
    # fwbp.generate_plots(op_dir,op_dir)


selected = folder_list


for slctd in selected:
    root_model_dir = os.getcwd()+os.sep+"Checkpoints"+os.sep+slctd    
    L = os.listdir(root_model_dir)
    for folder in L:
        m_dir = root_model_dir+os.sep+folder
        create_score_files_and_plots(m_dir,name)# Wroks fine only for five folded evaluaiton. -- #TODO need changes
















