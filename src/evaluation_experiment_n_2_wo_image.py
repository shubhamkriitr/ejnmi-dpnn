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
import performance_calculator as pc
from copy import deepcopy
import fold_wise_bar_plot as fwbp

#%%Select the network
from PXNET_SIGMOID_GAP import ProjectionNet as network

name = "PXNET_SIGMOID_GAP"
#%%Fetch data
break_pts = [(0,82), (83,111), (112,245)]
max_fold = 5


mn_v = 0.0#min value
mx_v = 92152.0#max value

data_ranges=[(0,82), (83,111), (112,245)]
#_, Y = data.get_parkinson_classification_data (ranges=data_ranges)
#_, X = data.get_parkinson_TF_data(ranges=data_ranges)
X , Y = data.get_new_dev_parkinson_cls_data(ranges=data_ranges)

X = X[:,:,:,:,0]
Y = Y[:,:,0]
X = (X-mn_v)/(mx_v-mn_v)

#%%For sorting out best models

model_dir = os.getcwd()+os.sep+"Checkpoints"
#0: keys for the folds for which search term has not been specified

#MAX_ACC
file_keys = {
#            "SET_":{1:["epoch_170","230023"],2:["epoch_220","231858"],
#                    3:["epoch_265","012938"],
#                      4:["epoch_130", "012938"],5:["epoch_125","011803"]}
                # "SET_":{1:["epoch_240", "215337"],# for selecting models from experiment 1s2
                #         2:["epoch_240", "220553"],
                #         3:["epoch_240", "221440"],
                #         4:["epoch_240", "222325"],
                #         5:["epoch_240", "223218"]
                #         }
                "SET_":{
                        1:["epoch_220","113617"],# for exp_n_2_try_1
                        2:["epoch_200","114652"],
                        3:["epoch_210","115732"],
                        4:["epoch_255","120804"],
                        5:["epoch_200","121838"],
                        
                }
        }
response = input("Select models?:(type `no` and `enter` to skip)"+str(file_keys))
if response.lower() == "yes":
    pc.pick_models(model_dir,file_keys,max_fold,search_level=3)
wait = input("wait-- press enter to proceed or Ctrl + C to abort.")

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
    fwbp.generate_plots(op_dir,op_dir)

selected = [#"Selected_2018-02-26-100557"
        #     "Selected_2018-02-28-144254-BEST-3-3"
        "Selected_2019-01-26-151956-exp_n_2_try_1",
    ]#Change it to the folder containing picked out models


for slctd in selected:
    root_model_dir = os.getcwd()+os.sep+"Checkpoints"+os.sep+slctd
    
    
    
    
    L = os.listdir(root_model_dir)
    for folder in L:
        m_dir = root_model_dir+os.sep+folder
        create_score_files_and_plots(m_dir,name)# Wroks fine only for five folded evaluaiton. -- #TODO need changes

















