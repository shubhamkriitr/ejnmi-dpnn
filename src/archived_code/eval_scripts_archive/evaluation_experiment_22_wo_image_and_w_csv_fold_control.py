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


def pick_models(model_dir,file_keys,folds_to_select=[1,2,3,4,5],search_level=3):
    op_dir =model_dir+os.sep+"Selected_"
    op_dir = ut.append_time_string(op_dir)
    
    os.mkdir(op_dir)
    for set_id in file_keys.keys():
        os.mkdir(op_dir+os.sep+set_id)
        for fold in folds_to_select:
            if fold in file_keys[set_id].keys():
                match_terms = deepcopy(file_keys[set_id][fold])
            else:
                match_terms = deepcopy(file_keys[set_id][0])
    
            match_terms.append("fold_"+str(fold))
            match_terms.append(set_id)
            ut.find_and_copy(model_dir,op_dir,set_id+os.sep+"fold_"+str(fold)+"_",match_terms,search_level,False)



#%%For sorting out best models

model_dir = os.getcwd()+os.sep+"Checkpoints"

folds_to_select = [5,]
file_keys = {
            
        "SET_21":{5:["epoch_318"],},
        }
response = input("Select models?:"+str(file_keys))
if response.lower() == "yes":
    pick_models(model_dir,file_keys,folds_to_select,search_level=3)
print("Done!")