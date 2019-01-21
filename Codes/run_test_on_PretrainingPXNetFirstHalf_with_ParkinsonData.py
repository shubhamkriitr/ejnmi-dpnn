#import run_PretrainingPXNetFirstHalf_fold_1
#import run_PretrainingPXNetFirstHalf_fold_2
#import run_PretrainingPXNetFirstHalf_fold_3
#import run_PretrainingPXNetFirstHalf_fold_4
#import run_PretrainingPXNetFirstHalf_fold_5

import tensorflow as tf
import logger
import data
import net
import numpy as np
import utility as ut
import os
from PretrainingPXNetFirstHalf import ProjectionNetFirstHalf
import matplotlib.pyplot as plt

CP_FOLDER = os.getcwd()+os.sep+"Checkpoints"+os.sep+"PXNET_FH_SET_1"
file_list = ["","PXNET_FH_fold_1_2018-01-31-205541_epoch_100_batch_50",
"PXNET_FH_fold_2_2018-01-31-205831_epoch_100_batch_50",
"PXNET_FH_fold_3_2018-01-31-210119_epoch_100_batch_50",
"PXNET_FH_fold_4_2018-01-31-210400_epoch_100_batch_50",
"PXNET_FH_fold_5_2018-01-31-210635_epoch_100_batch_50"]

log_list = []#add items to store in a log file
#%%SET PARAMS
note = "TESTING_PXNET_FISRT_HALF_"
log_list.append({"Note":note})

param = {"bsz":10
       }
fold = 5
max_fold = 5
param["max_fold"] = max_fold
param["fold"] = fold
param["model_path"] = CP_FOLDER+os.sep+file_list[fold]
if not  ut.does_exist(CP_FOLDER,file_list[fold]):
    os.mkdir(param["model_path"])
param["op_path"] = ut.append_time_string( CP_FOLDER+os.sep+file_list[fold]+os.sep+"Outputs_")
os.mkdir(param["op_path"])
summary_path = param["op_path"]
data_range = [[0,1076]]#parts of the dataset to  be used

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
print("List of vars that will be loaded:")
ut.print_sequence(comp_model.get_var_list())
log_list.append({"List of vars that will be loaded":comp_model.get_var_list()})


#%%Training

ut.create_log_file(summary_path+os.sep+"param_logs.txt",log_list)
with tf.Session(graph = comp_model.graph) as sess:
    tim = ut.Timer()
    tim.start()
    sess.run(tf.global_variables_initializer())
    comp_model.saver.restore(sess,param["model_path"])
    train_gen.reset_state_of_test_generator()
    val_gen.reset_state_of_test_generator()
    to_p = param["op_path"]+os.sep+"On_train"
    vo_p = param["op_path"]+os.sep+"On_val"
    os.mkdir(to_p)
    os.mkdir(vo_p)
    i = -1
    for idcs in  r:
        for j in range(idcs[0],idcs[1]+1):
            x_,y_,bsz = train_gen.get_next_test_batch(1)
            if bsz is None:
                break
            i+=1
            fn = to_p+os.sep+str(i)+"_train_vol_"+str(j)+".png"
            fna = fn[:-4]+"actual.png"
            fd = {comp_model.inputs[0]:x_}
            op_map = sess.run(comp_model.outputs[0], feed_dict = fd)
            plt.imsave(fn,op_map[0,:,:,0])
            plt.imsave(fna,y_[0,:,:,0])
            print(fn)
    i=-1
    for idcs in  s:
        for j in range(idcs[0],idcs[1]+1):
            x_,y_,bsz = val_gen.get_next_test_batch(1)
            if bsz is None:
                break
            i+=1
            fn = vo_p+os.sep+str(i)+"_val_vol_"+str(j)+".png"
            fna = fn[:-4]+"actual.png"
            plt.imsave(fna,y_[0,:,:,0])
            fd = {comp_model.inputs[0]:x_}
            op_map = sess.run(comp_model.outputs[0], feed_dict = fd)
            plt.imsave(fn,op_map[0,:,:,0])
            print(fn)
        
    
    
    