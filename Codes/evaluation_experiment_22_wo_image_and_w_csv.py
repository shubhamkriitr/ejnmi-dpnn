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
import fold_wise_bar_plot_and_csv as fwbp

#%%Select the network
from PXNET_SIGMOID_GAP import ProjectionNet as network

def pick_models(model_dir,file_keys,max_fold,search_level=3):
    op_dir =model_dir+os.sep+"Selected_"
    op_dir = ut.append_time_string(op_dir)
    
    os.mkdir(op_dir)
    for set_id in file_keys.keys():
        os.mkdir(op_dir+os.sep+set_id)
        for fold in range(1,max_fold+1):
            if fold in file_keys[set_id].keys():
                match_terms = deepcopy(file_keys[set_id][fold])
            else:
                match_terms = deepcopy(file_keys[set_id][0])
    
            match_terms.append("fold_"+str(fold))
            match_terms.append(set_id)
            ut.find_and_copy(model_dir,op_dir,set_id+os.sep+"fold_"+str(fold)+"_",match_terms,search_level,False)

name = "PXNET_SIGMOID_GAP"
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
            
        "SET_15":{0:["epoch_268"],},
        }
response = input("Select models?:"+str(file_keys))
if response.lower() == "yes":
    pc.pick_models(model_dir,file_keys,max_fold,search_level=3)
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
    pc.calculate_and_store_cf_metrics(ip_dir,op_dir,"score",has_image=False)
    fwbp.generate_plots(op_dir,op_dir)

selected = [#"Selected_2018-02-26-100557"
            #"Selected_2018-02-26-101300"
            #"Selected_2018-02-26-103532"#SET_2 max acc
            #"Selected_2018-02-26-110013-MINLS"#Min losssssssssss
            #"Selected_2018-02-26-111344-MINLS"#set-3 minls
            #"Selected_2018-02-26-112300-minls-6"
            #"Selected_2018-02-26-130657-3-maxac"
            #"Selected_2018-02-26-132811-7-ep135-minls"
            #"Selected_2018-02-26-143127-7-maxac"
            #"Selected_2018-02-26-171153-8-allep70"
            #"Selected_2018-02-26-184400-8-maxac-ep90"
            #"Selected_2018-02-27-185726-Set3-234054-ep140"
            #"Selected_2018-02-27-205208-BEST-1-minls",
            #"Selected_2018-02-27-205812-BEST-2-mix",
            "Selected_2018-02-27-223616-BEST-3",
            #"Selected_2018-02-27-224129-BEST-4",
            #"Selected_2018-02-28-143157-BEST-5"
            #"Selected_2018-02-28-144047-BEST-3-2"
            #"Selected_2018-02-28-144254-BEST-3-3-copy"
    ]#Change it to the folder containing picked out models


# for slctd in selected:
#     root_model_dir = os.getcwd()+os.sep+"Checkpoints"+os.sep+slctd
    
    
    
    
#     L = os.listdir(root_model_dir)
#     for folder in L:
#         m_dir = root_model_dir+os.sep+folder
#         create_score_files_and_plots(m_dir,name)

















