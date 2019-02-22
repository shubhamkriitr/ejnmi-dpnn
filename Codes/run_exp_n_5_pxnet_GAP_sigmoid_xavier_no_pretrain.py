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
----------
New EXT Dev Data specifications:
<class 'numpy.ndarray'>
(290, 95, 69, 79, 1)
float32
Mean 0.521783
Median 0.448994
Max 2.54515
Min -0.0328023
std.dev. 0.426738
----------
"""
mn_v = -0.0328023#min value
mx_v = 2.54515#max value 
max_fold = 5
save_step = 1 #num of epoch after which a checkpoint is saved
LRS = [1e-4, 1e-4, 1e-4, 1e-4 ]
regzr_scale =  [1e-5, 0.2*1e-5, 1e-6, 0.2*1e-6]
init_seeds = [109, 109, 109, 109]
initial_set_num = 18 # using 1000*(f^(-2))
#%%GPU CONFIG
gpu = 1
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True

wait = input ("wait...No pretrained models will be used for different folds.")
current_train_serial = -1
for lr_ in LRS:
    current_train_serial+=1
    set_id ="EXP_n_3_SET_"+str(initial_set_num)+"_"+str(lr_)+"_"
    set_id = ut.append_time_string(set_id)
    initial_set_num+=1

    #prepare intializer
    initializer = tf.contrib.layers.xavier_initializer(seed=init_seeds[current_train_serial])
    param = {
             "lr_classification":lr_, # lr_classification here means lr for all the model weights
             "bsz":10,
             "epoch_length":30,#NUM of batches that constitute an epoch,
             "epochs":400,
             "save_step":save_step,#num of epoch after which a checkpoint is saved
             "cost_fn":"INV_WEIGHTED_XENT",# -sigma(y_t*log(y_p))
             "initializer_seed":init_seeds[current_train_serial],
             "initializer":initializer,
             "reg_scale" : regzr_scale[current_train_serial],#change regularizer as well if you change this
             "regularizer" : tf.contrib.layers.l2_regularizer(scale=regzr_scale[current_train_serial])
           }

    for fold in range(1,max_fold+1):
        tf.reset_default_graph()
        log_list = []#add items to store in a log file
        #%%SET PARAMS
        note = "Training_PXNET_USING_new_extended_dataset with INV_WEIGHTED_XENT_cost"
        log_list.append({"Note":note})
        log_list.append({"Using min value=":mn_v, "Using max value=":mx_v})

        param["max_fold"] = max_fold
        param["fold"] = fold

        log_list.append(param)

        #%% SET UP FOLDERS AND DEVICE:
        CWD = os.getcwd()
        PD = os.path.abspath(os.pardir)

        model_name = "EXP_n_3_PXNET_GAP_SIG_NEWEXT_DSET"+set_id+"fold_"+str(fold)+"_"
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
        data_range_npd = [(0,90), (91,140), (141,289)]#parts of the new ext parkinson dataset to be used
        #fetch data copied from exp 20
        sp_pd = ut.get_split_ranges(data_range_npd,fold,max_fold)
        r_pd = sp_pd["train"]
        s_pd = sp_pd["val"]

        X_pd , Y_pd = data.get_newext_dev_parkinson_cls_data(ranges=r_pd)
        X_val_pd, Y_val_pd = data.get_newext_dev_parkinson_cls_data(ranges=s_pd)

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
                    "initializer":param["initializer"],
                    "regularizer":param["regularizer"]}
        #  compression model
        model = ProjectionNet(arg_dict)

        model.build_network()
        print("List of vars that will be saved:")
        ut.print_sequence(model.get_var_list())
        log_list.append({"List of vars that will be saved":model.get_var_list()})
        ##OPTIMIZER
        with model.graph.as_default():
            optimizer2 = tf.train.AdamOptimizer( learning_rate = param["lr_classification"])
            var_list_2 = model.get_var_list() # All the variables
            grads = tf.gradients(model.costs[0], var_list_2)
            grads2 = grads
            train_op2 = optimizer2.apply_gradients(zip(grads2,var_list_2))
            train_step = train_op2


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
            sess.run(tf.global_variables_initializer())
            #dummy=input("Wait")
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








