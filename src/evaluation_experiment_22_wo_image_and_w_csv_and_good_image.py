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
import performance_calculator_good_image as pc
from copy import deepcopy
import fold_wise_bar_plot_and_csv as fwbp

#%%Select the network
from PXNET_SIGMOID_GAP import ProjectionNet as network

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
            
            #"SET_7_":{0:["epoch_300"]}
            #"SET_2_":{0:["epoch_285"],3:["epoch_265"],
             #         4:["epoch_265"],5:["epoch_280"]}#MAXAC
            #"SET_2_":{0:["epoch_300"],2:["epoch_285"]}#MIN LOSS
            #"SET_3_":{1:["epoch_280"],2:["epoch_125"],3:["epoch_200"],
            #          4:["epoch_255"],5:["epoch_300"]}#minls
            #"SET_3_":{1:["epoch_280"],2:["epoch_290"],3:["epoch_290"],
            #         4:["epoch_300"],5:["epoch_300"]}
            #"SET_7_":{0:["epoch_135"]}
            #"SET_7_":{0:["epoch_260"],5:["epoch_240"]}
            #"SET_8_":{0:["epoch_70"]}#min_loss
            #"SET_8_":{0:["epoch_90"]}#max_acc
            #'SET_3_':{0:["epoch_140", "234054"]}
            #"SET_":{1:["epoch_210","121544"],2:["epoch_185","203442"],
                   # 3:["epoch_240","234054"],
                     # 4:["epoch_255", "234054"],5:["epoch_300","234054"]}
            
            #"SET_":{1:["epoch_160","230023"],2:["epoch_215","231858"],
                    #3:["epoch_260","012938"],
                     # 4:["epoch_255", "234054"],5:["epoch_300","234054"]}
            #"SET_":{1:["epoch_160","230023"],2:["epoch_215","231858"],
                    #3:["epoch_260","012938"],
                     # 4:["epoch_125", "012938"],5:["epoch_120","011803"]}#USED FOR PAPER
            
            #"SET_":{
                      #0:["epoch_125", "012938"]}
            
            #"SET_":{1:["epoch_160","230023"],2:["epoch_230","122023"],
                    #3:["epoch_260","012938"],
                     # 4:["epoch_125", "012938"],5:["epoch_120","011803"]}
            #"SET_":{1:["epoch_155","230023"],2:["epoch_210","231858"],
                    #3:["epoch_255","012938"],
                     # 4:["epoch_120", "012938"],5:["epoch_115","011803"]}
            #"SET_":{1:["epoch_170","230023"],2:["epoch_220","231858"],
                   # 3:["epoch_265","012938"],
                      #4:["epoch_130", "012938"],5:["epoch_125","011803"]}
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
    pc.calculate_and_store_cf_metrics(ip_dir,op_dir,"score",has_image=True)
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
            #"Selected_2018-02-27-223616-BEST-3",
            #"Selected_2018-02-27-224129-BEST-4",
            #"Selected_2018-02-28-143157-BEST-5"
            #"Selected_2018-02-28-144047-BEST-3-2"
            #"Selected_2018-02-28-144254-BEST-3-3-copy"
            "Selected_2018-02-27-223616-BEST-3-copy-For-good-projections"
    ]#Change it to the folder containing picked out models


for slctd in selected:
    root_model_dir = os.getcwd()+os.sep+"Checkpoints"+os.sep+slctd
    
    
    
    
    L = os.listdir(root_model_dir)
    for folder in L:
        m_dir = root_model_dir+os.sep+folder
        create_score_files_and_plots(m_dir,name)

















