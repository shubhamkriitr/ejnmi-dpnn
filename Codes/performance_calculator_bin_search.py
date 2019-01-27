#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 23:22:00 2018

@author: shubham
"""

import tensorflow as tf
import numpy as np
import h5py as hf
import os
import data
from data import DataGenerator
from copy import deepcopy
import utility as ut
print(type(np.int16))

import imp
import matplotlib.pyplot as plt
def get_srno_array(ranges,dtype=np.int16):
    """
    Returns an array of integers representing sr. no. corresponding
    to the intervals in ranges.

    Args:
        ranges(:obj:`list` of :obj:`list`): intervals
        dtype (:obj:`type`): data type

    Returns:
        An array of type `dtype`

    Examples:

        >>>print(get_srno_array([[1,2],[99,101]]))
        [[  1]
        [  2]
        [ 99]
        [100]
        [101]]
    """
    sz = 0
    for r in ranges:
        sz = sz + (r[1]-r[0] + 1)
    sr_arr = np.zeros(shape=(sz,1), dtype=dtype)
    i = 0
    for r in ranges:
        c_sz = (r[1]-r[0] + 1)# chunk size
        sr_arr[i:(i+c_sz),0] = np.linspace(r[0],r[1],c_sz)
        i = i + c_sz
    return sr_arr

# print(get_srno_array([[1,2],[99,100],[15,25]]))

def get_data_generators (fold_number,X_data,Y_data,break_pts):
    #DSET_LOC = data.DSET_FOLDER+os.sep+"P_DATA.h5"
    l = break_pts
    sp = ut.get_split_ranges(l,fold_number)
    tr = sp["train"]
    vl = sp["val"]
    X , Y = data.carve_out_chunk(X_data,Y_data,tr)
    X_val, Y_val = data.carve_out_chunk(X_data,Y_data,vl)

    print ("Input Shape",X.shape)
    print ("Val Input Shape",X_val.shape)


    train_serial = get_srno_array(tr,np.float32)
    val_serial = get_srno_array(vl,np.float32)
    return {"train":DataGenerator(X, Y, True),
            "val":DataGenerator(X_val, Y_val, True),"train_shape":X.shape,"val_shape":X_val.shape,
            "train_serial":train_serial,"val_serial":val_serial}

def evaluate (saved_model_path, output_file_path, model,fold, name,X,Y,break_pts):
    name = name+"_fold_"+str(fold)
    model.build_network()
    sess = tf.Session(graph=model.graph)
    dic = get_data_generators(fold,X,Y,break_pts)
    t_s = dic["train_shape"][0]
    v_s = dic["val_shape"][0]
    t_gen = dic["train"]
    v_gen = dic["val"]
    t_serial = dic["train_serial"]
    v_serial = dic["val_serial"]
    model.saver.restore(sess, saved_model_path)
    fn = output_file_path + os.sep + name+".h5"
    with hf.File(fn,"w") as f:
        grp = f.create_group("train")
        run_evaluation_steps(sess,model,t_s,t_gen,t_serial,grp)
        grp = f.create_group("val")
        run_evaluation_steps(sess,model,v_s,v_gen,v_serial,grp)
    sess.close()
    print("CLOSED")


def run_evaluation_steps (sess,model,sz,gen,serial,grp,batch_size=5):
    """Session model size, generator and hdf group """
    grp.create_dataset("serial",dtype=np.float32,data=serial)
    grp.create_dataset("outputs",shape=(sz,3),dtype=np.float32)
    grp.create_dataset("td_outputs",shape=(sz,3),dtype=np.float32)
    grp.create_dataset("labels", shape=(sz,3),dtype=np.float32)
    run_list = []
    #ORDER: Y Y_OP Y_TH PROJECTIONS
    run_list.append(model.labels[0])
    run_list.append(model.outputs[0])
    run_list.append(model.outputs[1])
    if len(model.extra_outputs)>0:
        grp.create_dataset("projections",shape=(sz,95,69,1),dtype=np.float32)
        run_list.append(model.extra_outputs["projections"])
    i = 0
    gen.reset_state_of_test_generator()
    while True:
        X, Y, bs = gen.get_next_test_batch(batch_size)
        if X is None:
            print("===OVER===")
            break
        print("From ",i,"to",i+bs,"="*10)
        fd = {model.inputs[0]:X,model.labels[0]:Y}
        if len(model.extra_outputs)>0:
            A,B,C,D = sess.run(run_list,feed_dict=fd)
            grp["labels"][i:i+bs] = A
            grp["outputs"][i:i+bs] = B
            grp["td_outputs"][i:i+bs] = C
            grp["projections"][i:i+bs] = D
            i = i+bs
        else:
            A,B,C = sess.run(run_list,feed_dict=fd)
            grp["labels"][i:i+bs] = A
            grp["outputs"][i:i+bs] = B
            grp["td_outputs"][i:i+bs] = C
            i = i+bs

def calculate_and_store_scores (input_folder, output_folder, name, net, model_arg_dict,X, Y,break_pts):
    """
    Args:
        break_pts(`list`): e.g. [(0,90), (91,120), (121,256)] parts of dataset to be considered separately for fetching data for different folds.
    """
    file_list = os.listdir(input_folder)
    for fold in range(1,6):
        sub_str = "fold_"+str(fold)
        for strs in file_list:
            if sub_str in strs:
                #imp.reload(tf)
                tf.reset_default_graph()
                model_graph = net(model_arg_dict)
                saved = input_folder+os.sep+strs
                strs = os.listdir(input_folder+os.sep+strs)
                for s in strs:
                    if "meta" in s:
                        strs = s
                        break
                strs = s[:-5]
                saved = saved+os.sep+strs
                print("--"*10,saved)
                #dummy = input("wait")
                evaluate(saved,output_folder,model_graph,fold,name,X,Y,break_pts)


def calculate_and_store_cf_metrics (input_folder, output_folder, suffix,has_image=False):
    file_list = os.listdir(input_folder)
    for fold in range(1,6):
        sub_str = "fold_"+str(fold)
        for strs in file_list:
            if sub_str in strs:
                ifn = input_folder + os.sep + strs
                ofn = output_folder + os.sep + strs[0:-3] + suffix + ".h5"
                print("Input Location:",ifn)
                print("Output Location:",ofn)
                f = hf.File(ifn,"r")
                g = hf.File(ofn,"w")
                for key in ["train","val"]:
                    print("Processing--",key)
                    grp = g.create_group(key)
                    cfmat = get_confusion_matrix(f[key+"/labels"][:],
                                                 f[key+"/td_outputs"][:],
                                                 np.float32)
                    _store_cfscores(grp,cfmat)
                if has_image:
                    img_loc = ofn[0:-3]
                    os.mkdir(img_loc)
                    for key in ["train","val"]:
                        img = f[key+"/projections"]
                        sr = f[key+"/serial"]
                        for i in range(img.shape[0]):
                            fig, ax = plt.subplots( nrows=1, ncols=1 )
                            ax.imshow(img[i,:,:,0])
                            fig.savefig(img_loc+os.sep+key+"_"+str(i)+"_"+str(sr[i])+".png")
                            plt.close(fig)    # close the figure
                f.close()
                g.close()



def _store_cfscores (grp,cf_mat,class_map={0:'MSA',
                                           1:'PSP',
                                           2:'PD'}):
    """
    Args:
        grp : hdf5 group object in output file
        cf_mat": confusion mat:
    """
    keys = ["TPR","TNR","PPV","NPV"]
    scores = {}
    print("=="*10)
    for k in class_map.keys():
        print("--"*10)
        scores[class_map[k]] = get_scores_from_cfmat(cf_mat[(int)(k)])
        print ("Class:",k,class_map[k],"Mat:\n",cf_mat[(int)(k)])
        grp2 = grp.create_group(class_map[k])
        arr = np.zeros((1,4),dtype=np.float32)
        for i in range(4):
            arr[0,i] = scores[class_map[k]][keys[i]]
        grp.create_dataset(class_map[k]+"_xls",dtype=np.float32,data=arr)
        grp.create_dataset(class_map[k]+"_cfmat",dtype=np.float32,data=cf_mat[(int)(k)])

        for key in keys:
            print("Key",key,"Score:",scores[class_map[k]][key])
            grp2.create_dataset(key,shape=(1,1),dtype=np.float32,data=scores[class_map[k]][key])
    print("Scores",scores)

def get_scores_from_cfmat (cfmat):
    TP = cfmat[0,0]
    FP = cfmat[0,1]
    FN = cfmat[1,0]
    TN = cfmat[1,1]
    TPR, TNR, PPV, NPV = 0,0,0,0
    if (TP + FN) > 0:
        TPR = TP/(TP + FN)
    if (FP + TN) > 0:
        TNR = TN/(FP + TN)
    if (TP + FP) > 0:
        PPV = TP/(TP + FP)
    if (TN + FN) > 0:
        NPV = TN/(TN + FN)
    return {"TPR":TPR,"TNR":TNR,"PPV":PPV,"NPV":NPV}

def get_confusion_matrix (y_true, y_pred, dtype=np.float32):
    """
    Returns class-wise confusion matrix.
    e.g. for three classes
        [[[TP1, FP1],
         [FN1, TN1]],
        [[TP2, FP2],
         [FN2, TN2]],
        [[TP3, FP3],
         [FN3, TN3]]]
    """
    assert (len(y_true.shape)==2 and y_true.shape == y_pred.shape)
    nch = y_true.shape[1]
    dsz = y_true.shape[0]
    mat = np.zeros(shape=(nch,2,2),dtype=dtype)
    print ("Dataset size", dsz)
    for i in range(dsz):
        for ch in range(nch):
            mat[ch,:,:] = mat[ch,:,:] + _get_cf(y_true[i,ch],y_pred[i, ch], dtype)
    return mat


def _get_cf (y_t,y_p,dtype=np.float32):
    """Returns a 2x2 matrix.
        e.g if y_t = 1 and y_p = 0 (i.e False Negative)
        then output is:
            [[0,0],
             [1,0]]
    """
#    if y_p > 0.5:
#        y_p = 0
#        print(type(y_p))
#    else:
#        y_p = 1
#    if y_t > 0.5:
#        y_t = 0
#        print(type(y_t))
#    else:
#        y_t = 1
    thd = 0.5
    ans = np.zeros(shape=(2,2),dtype=dtype)
    if y_p >= thd and y_t >= thd:
        ans[0,0]  = 1
    elif y_p >= thd and y_t < thd:
        ans[0,1]  = 1
    elif y_p < thd and y_t >= thd:
        ans[1,0]  = 1
    elif y_p < thd and y_t < thd:
        ans[1,1]  = 1
    else:
        raise AssertionError()
    return ans

def pick_models(model_dir,file_keys,max_fold,search_level=3,tag="", tagsep="tgsep_"):
    op_dir =model_dir+os.sep
    sufx = "Selected_"
    sufx = ut.append_time_string(sufx)
    sufx = sufx + tagsep+tag+ tagsep # tagging for listing the result in excel
    op_dir = op_dir + os.sep + sufx
    
    os.mkdir(op_dir)
    for set_id in file_keys.keys():
        os.mkdir(op_dir+os.sep+set_id)
        for fold in range(1,max_fold+1):
            if fold in file_keys[set_id].keys():
                match_terms = deepcopy(file_keys[set_id][fold])
            else:
                match_terms = deepcopy(file_keys[set_id][0])
    
            match_terms.append("fold_"+str(fold))
            match_terms.append(set_id)
            ut.find_and_copy(model_dir,op_dir,set_id+os.sep+"fold_"+str(fold)+"_",match_terms,search_level,False)
    return sufx


if __name__ == "__main__":
    input_folder= "/localhome/reuter/Desktop/Project/Master/7_Projection_Net_5_fold_cross_val/Code/Checkpoints/SET-2-Avg-Pooled-Eval"
    output_folder= "/localhome/reuter/Desktop/Project/Master/7_Projection_Net_5_fold_cross_val/Code/Checkpoints/SET-2-Avg-Pooled-Eval-Outputs"
    model_arg_dict = {"verbose":True, "input_shape":[None,95,69,79],
                  'weight_decay':1e-7, "cost_function": "XENT"}
    model_graph = net(model_arg_dict)
    name = "proj_net_avg_pool"
    calculate_and_store_scores (input_folder, output_folder, name,model_graph)

    ip_loc = "/home/shubham/Desktop/ENIGMA_CODES/Master/Outputs/scores-7/raw"
    op_loc = "/home/shubham/Desktop/ENIGMA_CODES/Master/Outputs/scores-7/processed"
    calculate_and_store_cf_metrics(ip_loc, op_loc, "score")










