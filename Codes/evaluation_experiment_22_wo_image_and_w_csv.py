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
            
        #     "SET_1":{3:["epoch_250"],
        #              2:["epoch_265"],
        #              0:["epoch_300"]}

        #  "SET_2":{1:["epoch_229"],
        #              4:["epoch_220"],
        #              5:["epoch_172"],
        #              0:["epoch_300"]}

        # "SET_5":{1:["epoch_176"], # SET - 5 - A
        #             2:["epoch_122"],
        #             3:["epoch_276"],
        #              4:["epoch_216"],
        #              5:["epoch_187"]}
        # "SET_7":{1:["epoch_154"], # SET - 7 -A
        #             2:["epoch_154"],
        #             3:["epoch_165"],
        #              4:["epoch_235"],
        #              5:["epoch_145"]}
        # "SET_7":{1:["epoch_144"], # SET - 7 - B
        #             2:["epoch_142"],
        #             3:["epoch_222"],
        #              4:["epoch_185"],
        #              5:["epoch_142"]}
        # "SET_5":{1:["epoch_183"], # SET - 5 -B
        #             2:["epoch_124"],
        #             3:["epoch_283"],
        #              4:["epoch_228"],
        #              5:["epoch_200"]}
        # "SET_5":{1:["epoch_122"], # SET - 5SA
        #             2:["epoch_122"],
        #             3:["epoch_192"],# not that good
        #              4:["epoch_127"],
        #              5:["epoch_150"]}# not that good
        # "SET_7":{1:["epoch_122"], # SET - 7SA
        #             2:["epoch_122"],
        #             3:["epoch_150"],# not that good
        #              4:["epoch_127"],
        #              5:["epoch_126"]}
        # "SET_10":{1:["epoch_266"], # SET - 10A
        #             2:["epoch_225"],
        #             3:["epoch_287"],# 
        #              4:["epoch_289"],
        #              5:["epoch_238"]}
        # "SET_10":{1:["epoch_163"], # SET - 10B
        #             2:["epoch_130"],
        #             3:["epoch_210"],# 
        #              4:["epoch_200"],
        #              5:["epoch_90"]},# remove it
        # "SET_10":{1:["epoch_240"], # SET - 10C
        #             2:["epoch_150"],
        #             3:["epoch_280"],# 
        #              4:["epoch_250"],
        #              5:["epoch_90"]},# remove it
        # "SET_11":{1:["epoch_254"], # SET - 11A
        #             2:["epoch_264"],
        #             3:["epoch_255"],# 
        #              4:["epoch_246"],
        #              5:["epoch_162"]},
        # "SET_11":{1:["epoch_134"], # SET - 11B
        #             2:["epoch_103"],
        #             3:["epoch_210"],# 
        #              4:["epoch_90"],#remove
        #              5:["epoch_90"]},# remove
        # "SET_11":{1:["epoch_150"], # SET - 11C
        #             2:["epoch_103"],
        #             3:["epoch_250"],# 
        #              4:["epoch_90"],# remove
        #              5:["epoch_90"]},#remove
        # "SET_11":{1:["epoch_254"], # SET - 11D
        #             2:["epoch_288"],
        #             3:["epoch_269"],# 
        #              4:["epoch_258"],
        #              5:["epoch_177"]},
        # "SET_7":{1:["epoch_154"], # NSET - 7 -A
        #             2:["epoch_154"],
        #             3:["epoch_165"],
        #              4:["epoch_235"],
        #              5:["epoch_188"]},
        # "SET_5":{1:["epoch_289"], # NSET - 5 -B
        #             2:["epoch_278"],
        #             3:["epoch_296"],
        #              4:["epoch_288"],
        #              5:["epoch_237"]},
        "SET_7":{1:["epoch_210"], # NSET - 7 -B
                    2:["epoch_212"],
                    3:["epoch_208"],
                     4:["epoch_234"],
                     5:["epoch_200"]},
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

















