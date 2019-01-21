#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 22:11:52 2018
Running PXNet-Fisrt Half for pretraining of PXNet-Full
@author: shubham
"""
import tensorflow as tf
import logger
import data
import net
import numpy as np
import utility as ut
import os
from PretrainingPXNetFirstHalf import ProjectionNetFirstHalf


tf.reset_default_graph()
log_list = []#add items to store in a log file
#%%SET PARAMS
note = "RUNNING_PXNET_FISRT_HALF_"
log_list.append({"Note":note})

param = {"lr":0.00001,
         "bsz":10,
         "epoch_length":50,#NUM of batches that constitute an epoch,
         "epochs":100,
         "save_step":5,#num of epoch after which a checkpoint is saved
       }

data_range = [[0,1076]]#parts of the dataset to  be used
fold = 5
max_fold = 5
param["max_fold"] = max_fold
param["fold"] = fold

log_list.append(param)

#%% SET UP FOLDERS AND DEVICE:
CWD = os.getcwd()
PD = os.path.abspath(os.pardir)

model_name = "PXNET_FH_NORM"+"fold_"+str(fold)+"_"
model_name = ut.append_time_string(model_name)

if not ut.does_exist(CWD,"Checkpoints"):
    os.mkdir(CWD+os.sep+"Checkpoints")

model_folder = ut.add_file_or_folder(CWD +os.sep+ "Checkpoints",
                                     model_name,True)#adds a folder dedicated to the  current model

log_list.append({"model_name":model_name})


if not ut.does_exist(CWD,"Summaries"):
    os.mkdir(CWD+os.sep+"Summaries")

summary_path = ut.add_file_or_folder(CWD+os.sep+"Summaries",
                                     model_name,True)#adds a folder dedicated to the  current model
log_list.append({"summary_path":summary_path})

#%%DATA
#fetch data
l = data_range
sp = ut.get_split_ranges(l,fold,max_fold)
r = sp["train"]
s = sp["val"]

X , Y = data.get_pretraining_TF_data(ranges=r)
print ("Input Shape",X.shape)
log_list.append({"Input Shape":X.shape})
X = X[:,:,:,:,0]

X_val, Y_val = data.get_pretraining_TF_data(ranges=s)
X_val = X_val[:,:,:,:,0]
#mormalize
X = X/95618
X_val = X_val/95618

print ("Val Input Shape",X_val.shape)
log_list.append({"Val Input Shape":X_val.shape})


log_list.append({"training on":r, 'validating_on':s})
log_list.append({"X":X, "Y":Y, "X_val":X_val, "Y_val":Y_val})

train_gen = data.DataGenerator(X,Y)
val_gen = data.DataGenerator(X_val,Y_val)

#%%Create model
#  may add elements to arg_dict to adjust regularizers etc.
arg_dict = {}
#  compression model
comp_model = ProjectionNetFirstHalf(arg_dict)

comp_model.build_network()
print("List of vars that will be saved:")
ut.print_sequence(comp_model.get_var_list())
log_list.append({"List of vars that will be saved":comp_model.get_var_list()})
##OPTIMIZER
with comp_model.graph.as_default():
    optimizer = tf.train.AdamOptimizer( learning_rate = param["lr"])
    train_step = optimizer.minimize(comp_model.costs[0])

#%%Training

logger_args =  {"scope": "LOGS",
                "model": comp_model ,
                "session":None ,
                "log_root_folder":summary_path,
                "bsz":param["bsz"],
                "train_gen":train_gen,
                "val_gen":val_gen}
log_list.append(logger_args)
lgr = logger.Logger(logger_args)

with comp_model.graph.as_default():
    lgr.create_summary_ops(comp_model.node_dict,{},comp_model.histogram_list)

file_name = model_folder + os.sep + model_name

ut.create_log_file(summary_path+os.sep+"param_logs.txt",log_list)
with tf.Session(graph = comp_model.graph) as sess:
    tim = ut.Timer()
    tim.start()
    lgr.set_session(sess)
    sess.run(tf.global_variables_initializer())
    for epoch in range(1,param["epochs"]+1):
        for batch in range(1,param["epoch_length"]+1):
            x_, y_ = train_gen.get_next_batch(param["bsz"])
            fd = {comp_model.inputs[0]:x_, comp_model.true_outputs[0]:y_}
            loss, _ = sess.run([comp_model.costs[0],train_step],feed_dict=fd)
            log = "Epoch:{} Batch:{} Loss:{} Time:".format(epoch,batch,loss)+tim.elapsed()
            print(log,"\n"+"-"*len(log))
        lgr.record_train_summary(epoch)
        lgr.record_val_summary(epoch)
        if epoch%param["save_step"]==0:
            save_as = file_name + "_epoch_{}_batch_{}".format(epoch,batch)
            print("saving_check_point:",save_as)
            comp_model.saver.save(sess,save_as)








