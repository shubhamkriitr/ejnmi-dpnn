#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 22:11:52 2018
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
New training data specifications:
<class 'numpy.ndarray'>
(246, 95, 69, 79, 1)
float32
Mean 8041.11
Median 6372.97
Max 92152.0
Min 0.0
std.dev. 7255.93
----------
"""
mn_v = 0.0#min value
mx_v = 92152.0#max value
max_fold = 5
save_step = 1 #num of epoch after which a checkpoint is saved
initializer = tf.contrib.layers.xavier_initializer(seed=108)
LRS = [(1e-5, 1e-4)]
initial_set_num = 12 # using 1000*(f^(-2))#11 # same as 10#10#using 100*f^(-2)#9 #8 - for this fold 4/5 had bad val characteristic#set-8 f^(-2) as weights #7# SET-3, 4, and 6 are not valid
#%%GPU CONFIG
gpu = 1
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
#%%SLECTING MODELS FOR LOADING WEIGHTS
saved_folder = "PRETRAINED_COMPRESSION_MODELS"
paths = {}
for fd in range(1,max_fold+1):
    match_terms = ["fold_"+str(fd)+"_",".meta"]
    path = ut.find_paths(os.getcwd()+os.sep+saved_folder,match_terms,level=3)
    print(fd, path)
    assert(len(path)==1)
    paths[fd]=path[0][:-5]

print("paths",paths)
wait = input ("wait...Verify the models that will be used for different folds.")
#TODO_ may want to sepcify gpu memory allocation type and GPU to use.
for lrs in LRS:
    set_id ="EXP_n_3_SET_"+str(initial_set_num)+"_"+str(lrs)+"_"
    set_id = ut.append_time_string(set_id)
    initial_set_num+=1

    param = {"lr_compression":lrs[0],
             "lr_classification":lrs[1],
             "bsz":10,
             "epoch_length":30,#NUM of batches that constitute an epoch,
             "epochs":300,
             "save_step":save_step,#num of epoch after which a checkpoint is saved
             "cost_fn":"INV_WEIGHTED_XENT",# -sigma(y_t*log(y_p))
             "initializer":initializer
           }

    for fold in range(1,max_fold+1):
        tf.reset_default_graph()
        log_list = []#add items to store in a log file
        #%%SET PARAMS
        note = "FINE_TUNING_PXNET_USING_PRETRAINED_COMPRESSION_PART_OF_exp_n_1s2_using_INV_WEIGHTED_XENT_cost"
        log_list.append({"Note":note})
        log_list.append({"Using min value=":mn_v, "Using max value=":mx_v})

        pretrained_model = paths[fold]

        print(pretrained_model)
        param["model_location"]=pretrained_model

        param["max_fold"] = max_fold
        param["fold"] = fold

        log_list.append(param)

        #%% SET UP FOLDERS AND DEVICE:
        CWD = os.getcwd()
        PD = os.path.abspath(os.pardir)

        model_name = "EXP_n_3_PXNET_GAP_SIG_FINE_TUNED"+set_id+"fold_"+str(fold)+"_"
        model_name = ut.append_time_string(model_name)

        if not ut.does_exist(CWD,"Checkpoints"):
            os.mkdir(CWD+os.sep+"Checkpoints")


        set_cp_folder = CWD +os.sep+ "Checkpoints"+os.sep+set_id
        if not os.path.exists(set_cp_folder):
            os.mkdir(set_cp_folder)

        model_folder = set_cp_folder + os.sep+model_name#adds a folder dedicated to the  current model
        if not os.path.exists(model_folder):
            os.mkdir(model_folder)

        log_list.append({"model_name":model_name})


        if not ut.does_exist(CWD,"Summaries"):
            os.mkdir(CWD+os.sep+"Summaries")

        set_sm_folder = CWD+os.sep+"Summaries"+os.sep+set_id

        if not os.path.exists(set_sm_folder):
            os.mkdir(set_sm_folder)


        summary_path = set_sm_folder + os.sep + model_name
        if not os.path.exists(summary_path):
            os.mkdir(summary_path)

        log_list.append({"summary_path":summary_path})

        #%%DATA
        #fetch data
        data_range_npd = [(0,82), (83,111), (112,245)]#parts of the new parkinson dataset to be used
        #fetch data copied from exp 20
        sp_pd = ut.get_split_ranges(data_range_npd,fold,max_fold)
        r_pd = sp_pd["train"]
        s_pd = sp_pd["val"]

        X_pd , Y_pd = data.get_new_dev_parkinson_cls_data(ranges=r_pd)
        X_val_pd, Y_val_pd = data.get_new_dev_parkinson_cls_data(ranges=s_pd)

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
        arg_dict = {"cost_function":param["cost_fn"],
                    "initializer":param["initializer"]}
        #  compression model
        model = ProjectionNet(arg_dict)

        model.build_network()
        print("List of vars that will be saved:")
        ut.print_sequence(model.get_var_list())
        log_list.append({"List of vars that will be saved":model.get_var_list()})
        ##OPTIMIZER
        with model.graph.as_default():
            optimizer1 = tf.train.AdamOptimizer( learning_rate = param["lr_compression"])
            optimizer2 = tf.train.AdamOptimizer( learning_rate = param["lr_classification"])


            var_list_1 = model.compression_part_var_list
            var_list_2 = model.classification_part_var_list
            grads = tf.gradients(model.costs[0], var_list_1 + var_list_2)
            grads1 = grads[:len(var_list_1)]
            grads2 = grads[len(var_list_1):]

            train_op1 = optimizer1.apply_gradients(zip(grads1,var_list_1))
            train_op2 = optimizer2.apply_gradients(zip(grads2,var_list_2))
            train_step = tf.group(train_op1, train_op2)


        #%%Training

        logger_args =  {"scope": "LOGS",
                        "model": model ,
                        "session":None ,
                        "log_root_folder":summary_path,
                        "bsz":param["bsz"],
                        "train_gen":train_gen,
                        "val_gen":val_gen}
        log_list.append(logger_args)
        lgr = logger.Logger(logger_args)

        with model.graph.as_default():
            lgr.create_summary_ops(model.node_dict,{},model.histogram_list)

        file_name = model_folder + os.sep + model_name

        ut.create_log_file(summary_path+os.sep+"param_logs.txt",log_list)
        with tf.Session(graph = model.graph,
                        config=config) as sess:
            tim = ut.Timer()
            tim.start()
            lgr.set_session(sess)
            wt_tnsr = model.graph.get_tensor_by_name("PXNet/block1/W:0")
            sess.run(tf.global_variables_initializer())
            wt = sess.run(wt_tnsr)
            ut.get_array_info(wt,"WT before loading")
            #dummy = input("See Wt Val.")

            model.compression_part_saver.restore(sess,pretrained_model)

            wt = sess.run(wt_tnsr)
            ut.get_array_info(wt,"After Loading")
            #dummy=input("Wait")
            print("Loaded:",pretrained_model,"for fold {}.".format(fold))
            for epoch in range(1,param["epochs"]+1):
                for batch in range(1,param["epoch_length"]+1):
                    x_, y_ = train_gen.get_next_batch(param["bsz"])
                    fd = {model.inputs[0]:x_, model.true_outputs[0]:y_}
                    accu,loss, _ = sess.run([model.accuracy,model.costs[0],train_step],feed_dict=fd)
                    log = "Epoch:{} Batch:{} Loss:{} Accuracy:{} Time:".format(epoch,batch,loss,accu)+tim.elapsed()
                    print(log,"\n"+"-"*len(log))
                lgr.record_train_summary(epoch)
                lgr.record_val_summary(epoch)
                if epoch%param["save_step"]==0 and epoch>50:
                    save_as = file_name + "_epoch_{}_batch_{}".format(epoch,batch)
                    print("saving_check_point:",save_as)
                    model.saver.save(sess,save_as)

        del X
        del Y
        del X_val
        del Y_val
        del model
        del lgr








