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
from PXNET_SIGMOID_GAP import ProjectionNet
"""
train specifications:
<class 'numpy.ndarray'>
(1077, 95, 69, 79)
float32
Mean 10282.7
Median 7646.11
Max 95617.8
Min -3173.77
std.dev. 9765.27
----------
----------
Y_whole specifications:
<class 'numpy.ndarray'>
(1334, 95, 69, 1)
float32
Mean 4.994355
Median 5.865527
Max 11.497833
Min -0.13825107
std.dev. 2.7348104

"""
gpu = 1
#print("available_devices",available_devices)
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)


mn_v = -3173.77#min value
mx_v = 95617.8#max value


max_fold = 5

data_range_pd = [(0,90), (91,120), (121,256)]#parts of the parkinsond dataset to be used
initial_set = 7
#lrs = [1e-3,5*1e-4,0.0001,5*1e-5,1e-5]
###Till SET 3 has been done
lrs = [1e-4]

set_num = initial_set-1


for lr in lrs:
    set_num+=1
    set_id =ut.append_time_string( "_SET_{}_".format(set_num))+"_"
    
    for fold in range(1,max_fold+1):
        tf.reset_default_graph()
        log_list = []#add items to store in a log file
        #%%SET PARAMS
        note = "EXP_20: Running new PXNET (GAP) using only Parkinson Data\
        Without any pretraining. sigmoid activated 2d map block"
        log_list.append({"Note":note})
        
        param = {"lr":lr,
                 "bsz":10,
                 "epoch_length":30,#NUM of batches that constitute an epoch,
                 "epochs":300,
                 "save_step":1,#num of epoch after which a checkpoint is saved
               }
        
        param["max_fold"] = max_fold
        param["fold"] = fold
        
        log_list.append(param)
        
        #%% SET UP FOLDERS AND DEVICE:
        CWD = os.getcwd()
        PD = os.path.abspath(os.pardir)
        
        model_name = "EXP_20_"+set_id+"fold_"+str(fold)+"_"
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
        sp_pd = ut.get_split_ranges(data_range_pd,fold,max_fold)
        r_pd = sp_pd["train"]
        s_pd = sp_pd["val"]
        
        X_pd , Y_pd = data.get_parkinson_classification_data(ranges=r_pd)
        X_val_pd, Y_val_pd = data.get_parkinson_classification_data(ranges=s_pd)
        
        X_pd = X_pd[:,:,:,:,0]
        X_val_pd = X_val_pd[:,:,:,:,0]
        
        X = X_pd
        Y = Y_pd[:,:,0]
        X_val = X_val_pd
        Y_val = Y_val_pd[:,:,0]
        
        print ("Val Input Shape",X_val.shape)
        log_list.append({"Val Input Shape":X_val.shape})
        log_list.append({"training on":r_pd, 'validating_on':s_pd})
        log_list.append({"X":X, "Y":Y, "X_val":X_val, "Y_val":Y_val})
        
        
        #mormalize
        X = (X-mn_v)/(mx_v-mn_v)
        X_val = (X_val-mn_v)/(mx_v-mn_v)
        train_gen = data.DataGenerator(X,Y)
        val_gen = data.DataGenerator(X_val,Y_val)
        
        #%%Create model
        #  may add elements to arg_dict to adjust regularizers etc.
        arg_dict = {}
        #  compression model
        comp_model = ProjectionNet(arg_dict)
        #wait = input("Before build")
        comp_model.build_network()
        #wait = input("wait: after build")
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
        
        config = tf.ConfigProto(log_device_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(graph = comp_model.graph,
                        config=config) as sess:
            #wait = input("wait: session start")
            tim = ut.Timer()
            tim.start()
            lgr.set_session(sess)
            sess.run(tf.global_variables_initializer())
            #wait = input("wait: after init")
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
        
        del X
        del Y
        del X_val
        del Y_val
        del comp_model
        del lgr








