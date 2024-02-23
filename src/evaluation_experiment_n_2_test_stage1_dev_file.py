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
import test_performance_calculator as tpc
from copy import deepcopy
import time

gpu = 1# using 2nd GPU
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

#%%Select the network
from PXNET_SIGMOID_GAP import ProjectionNet as network

## fetching the key map used below
training_idx_to_name = '/home/abhijit/nas_drive/Abhijit/Shubham/ejnmmi-dpnn/Codes/new_ext_train_data_idx_to_vol_mapping.txt'
    
with open(training_idx_to_name, 'r') as f:
    idx_to_name = eval(f.read())
#%%Fetch data
print("-----*****Have you changed Min-Max values?*****-----")
mn_v = -0.0328023#min value
mx_v = 2.54515#max value 
num_test_samples = 290 # running on train data
# mx_v =  41896.2 # Actual min & max of test dataset
# Min 0.0
dummy_wait = input("MAX_VALUE being used:"+str(mx_v))
# data_ranges=[(0,62),] Using default
X , _ = data.get_newext_dev_parkinson_cls_data(ranges=[[0, 289]])
data_idx_key_map = idx_to_name
X = X[:,:,:,:,0]
print("-----*****Change Min-Max values*****-----")
X = (X-mn_v)/(mx_v-mn_v)
## EDIT these identifiers
set_id_with_selection_id = "SET_15_18_19_20_21_train_data"#'your folder name should be TEST_ROOT_<set_id_with_selection_id>'
test_exp_identifier = "PXNET_GAP_SIG_WTD_NP_TEST_RESULT_on_training_data"+set_id_with_selection_id

input_folder = "/home/abhijit/nas_drive/Abhijit/Shubham/ejnmmi-dpnn/Codes/Checkpoints/TEST_ROOT_"+set_id_with_selection_id
op_folder = input_folder+os.sep+"OUTPUTS"
combined_op_folder = input_folder+os.sep+"ENSEMBLE_OUTPUT"
if not os.path.exists(op_folder):
    os.makedirs(op_folder)

if not os.path.exists(combined_op_folder):
    os.makedirs(combined_op_folder)

# tpc.calculate_and_store_test_predictions(input_folder, op_folder,test_exp_identifier,
#                                     net=network,model_arg_dict={},X=X,suffix="test_predns")
h5list = tpc._get_h5_file_list_to_combine(op_folder,[test_exp_identifier],1)
tpc.create_mapping(h5list, combined_op_folder+os.sep+"h5map.txt", op_folder)

tpc.combine_predictions_from_h5_file(op_folder, combined_op_folder, idx_vol_key_map=data_idx_key_map, num_rows=num_test_samples,
                                    model_name=test_exp_identifier, suffix="_predictions",match_terms=[], level=1)













