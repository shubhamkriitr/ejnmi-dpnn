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
from PretrainingPXNetFirstHalfSigmoid import ProjectionNetFirstHalf as model_class

name = "EXP_21_STAGE_II"
#%%Fetch data
break_pts = [(0,90), (91,120), (121,256)]
max_fold = 5


#%%For sorting out best models

model_dir = os.getcwd()+os.sep+"Checkpoints"
file_keys = {
            "STAGEII_4_":{0:["epoch_250"],3:["epoch_245"]}
        
        }

#pc.pick_models(model_dir,file_keys,max_fold,search_level=2)
wait = input('wait')

#%% Generate Maps
"""
Folder structure must be like this:
    MODEL_ROOT_FOLDER:
        SET_1:
            FOLD_1:
                model_fold_1.meta
                model_fold_1.data...
                model_fold_1.index
            FOLD_2:
                ...
        SET_2:
            ...
"""

import matplotlib.pyplot as plt
import tensorflow as tf
mn_v = -3173.77#min value
mx_v = 95617.8#max value

tf_mx = 11.497833#maxs_value for TF_DATA
tf_mn = -0.13825107


def convert_to_2d (Y):
    return Y[0,:,:,0]

def save_outputs (model_root_dir,model_class,data_range,data_fetcher,converter_fn,max_fold = 5):#  it is model EXPERIMENT SPECIFIC function#modify it as per use case
    root = model_root_dir
    dirs = os.listdir(model_root_dir)
    root_op_dir = ut.append_time_string( 
            os.path.abspath(os.path.join(root,os.pardir))+"_Outputs_")
    os.mkdir(root_op_dir)
    
    for set_ in dirs:
        model_dir = root+os.sep+set_
        model_op_dir = root_op_dir+os.sep+set_
        os.mkdir(model_op_dir)
        for fold in range(1,max_fold+1):
            files = ut.find_paths(model_dir,["fold_"+str(fold)+"_","meta"],level=2)
            assert(len(files)==1)# should be only one file for one fold
            file = files[0][:-5]#remove .meta
            op_dir = model_op_dir+os.sep+"fold_"+str(fold)+"_"
            os.mkdir(op_dir)
            #DATA
            #fetch data
            l = data_range
            sp = ut.get_split_ranges(l,fold,max_fold)
            r = sp["train"]
            s = sp["val"]
            
            X , Y = data_fetcher(ranges=r)
            X = X[:,:,:,:,0]
            X_val, Y_val = data_fetcher(ranges=s)
            X_val = X_val[:,:,:,:,0]
            #mormalize
            X = (X-mn_v)/(mx_v-mn_v)
            X_val = (X_val-mn_v)/(mx_v-mn_v)
            
            
            Y = (Y-tf_mn)/(tf_mx-tf_mn)
            Y = np.clip(Y,1e-7,1.0-1e-7)
            
            Y_val = (Y_val-tf_mn)/(tf_mx-tf_mn)
            Y_val = np.clip(Y_val,1e-7,1.0-1e-7)
            save_tf_maps(file,model_class(),op_dir,X,Y,r,X_val,Y_val,s,converter_fn)
        
        
    



def save_tf_maps(saved_model_path,model,op_dir, X_train,Y_train,train_ranges,X_val,Y_val,val_ranges,converter_fn):
    """train_ranges and val_ranges denote actual index at which those samples are stored."""
    root =  saved_model_path
    tf.reset_default_graph()#cleqar already existing graph
    model.build_network()
    sess = tf.Session(graph=model.graph)
    model.saver.restore(sess,saved_model_path)
    #op_folder = ut.append_time_string( os.path.abspath(os.path.join(root,os.pardir))+"_")
    
    trn_op_dir = op_dir+os.sep+"train"
    os.mkdir(trn_op_dir)
    _predict_and_save(trn_op_dir,model,sess,X_train,Y_train,train_ranges,converter_fn,"train")
    
    val_op_dir = op_dir+os.sep+"val"
    os.mkdir(val_op_dir)
    _predict_and_save(val_op_dir,model,sess,X_val,Y_val,val_ranges,converter_fn,"val")
    
    


def _predict_and_save (op_folder,model,sess,X,Y,ranges,converter_fn,suffix):
    #op_folder = ut.append_time_string( os.path.abspath(os.path.join(root,os.pardir))+"_")
    #os.mkdir(op_folder)
    
    i = -1
    for rng in ranges:
        for j in range(rng[0],rng[1]+1):
            i+=1
            fd = {model.inputs[0]:X[i:i+1]}
            output = sess.run(model.outputs[0], feed_dict = fd)
            output = converter_fn(output)
            actual = converter_fn(Y[i:i+1])
            pred_path = op_folder+os.sep+str(i)+"_"+str(j)+"_pred_"+suffix+".png"
            true_path = op_folder+os.sep+str(i)+"_"+str(j)+"_true_"+suffix+".png"
            plt.imsave(pred_path,output)
            plt.imsave(true_path,actual)
    
#%%RUN 
#some constants are defined in the beginnning of the file.
data_range = [[0,1076]]#parts of the dataset to  be used #[[0,256]]
data_fetcher = data.get_pretraining_TF_data

data_range = [(0,90), (91,120), (121,256)]
data_fetcher = data.get_parkinson_TF_data

root_dir = os.getcwd()+os.sep+"Checkpoints"+os.sep+"Selected_2018-02-25-224522-STAGEII_4"
save_outputs( root_dir,model_class,data_range,data_fetcher,convert_to_2d)
    
    

















