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

name = "PXNET_GAP"
#%%Fetch data
break_pts = [(0,90), (91,120), (121,256)]
max_fold = 5


mn_v = -3173.77#min value
mx_v = 95617.8#max value

data_ranges=[(0,90), (91,120), (121,256)]
#_, Y = data.get_parkinson_classification_data (ranges=data_ranges)
#_, X = data.get_parkinson_TF_data(ranges=data_ranges)
X , Y = data.get_parkinson_classification_data(ranges=data_ranges)

X = X[:,:,:,:,0]
Y = Y[:,:,0]
X = (X-mn_v)/(mx_v-mn_v)

#%%For sorting out best models

model_dir = os.getcwd()+os.sep+"Checkpoints"
#0: keys for the folds for which search term has not been specified

#MAX_ACC
#file_keys = {
#            "SET_7":{1:["epoch_250"], 2:["epoch_275"],3:["epoch_290"],4:["epoch_293"],5:["epoch_300"]}
#      
#       }

#MIN_LOSS
#file_keys = {
#            
#            "SET_7":{1:["epoch_285"], 2:["epoch_285"],3:["epoch_180"],4:["epoch_280"],5:["epoch_275"]}
#        
#        }

#MAX_ACC
file_keys = {
            
            "SET_7":{0:["epoch_263"], 2:["epoch_202"],3:["epoch_222"],4:["epoch_280"],5:["epoch_275"]}
        
        }

#pc.pick_models(model_dir,file_keys,max_fold,search_level=2)
wait = input("wait")

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
    pc.calculate_and_store_cf_metrics(ip_dir,op_dir,"score",has_image=True)
    fwbp.generate_plots(op_dir,op_dir)

selected = [#"Selected_2018-02-25-104756-MIN-LOSS",

	#"Selected_2018-02-25-104443-MAX-ACC"
    "Selected_2018-02-25-113650-MAX-ACC"
    ]#Change it to the folder containing picked out models


for slctd in selected:
    root_model_dir = os.getcwd()+os.sep+"Checkpoints"+os.sep+slctd
    
    
    
    
    L = os.listdir(root_model_dir)
    for folder in L:
        m_dir = root_model_dir+os.sep+folder
        create_score_files_and_plots(m_dir,name)

















