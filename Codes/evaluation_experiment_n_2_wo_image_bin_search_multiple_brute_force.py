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
#%%GPU CONFIG
gpu = 1
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
name = "PXNET_SIGMOID_GAP_NP"
#%%Fetch data
break_pts = [(0,90), (91,140), (141,289)]#parts of the new ext parkinson dataset to be used
max_fold = 5


mn_v = -0.0328023#min value
mx_v = 2.54515#max value
_ = input("Using min:{} and max:{}. Press emter to proceed".format(mn_v, mx_v))

data_ranges=[(0,90), (91,140), (141,289)]#parts of the new ext parkinson dataset to be used
#_, Y = data.get_parkinson_classification_data (ranges=data_ranges)
#_, X = data.get_parkinson_TF_data(ranges=data_ranges)
X , Y = data.get_newext_dev_parkinson_cls_data(ranges=data_ranges)

X = X[:,:,:,:,0]
Y = Y[:,:,0]
X = (X-mn_v)/(mx_v-mn_v)

#%%For sorting out best models

model_dir = os.getcwd()+os.sep+"Checkpoints"
#0: keys for the folds for which search term has not been specified

#MAX_ACC
fl = []

for eps in range(100, 301, 1):
    dt  = {"SET_20":{
                        1:["epoch_"+str(eps)],# for exp_n_2_try_2 
                        2:["epoch_"+str(eps)],
                        3:["epoch_"+str(eps)],
                        4:["epoch_"+str(eps)],
                        5:["epoch_"+str(eps)],
                        
            }}
    fl.append(dt)
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
        fldr = pc.pick_models(model_dir,f_keys,max_fold,search_level=3,tag=tag,tagsep="_tgx_")
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
















