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
import imp
import matplotlib.pyplot as plt
import pandas as pd

#Additional importds for testing

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


#%% Functions for evaluation on test dataset
def get_model_list_to_run (root_dir, extra_match_terms=[], search_at_level = 4):
    match_terms = [".meta"]
    for terms in extra_match_terms:
        match_terms.append(terms)
    L = ut.find_paths(root_dir, match_terms, level=search_at_level)
    L.sort()
    print(L)
    for i in range(len(L)):
        L[i] = L[i][:-5]#removing .meta
    return L

def create_mapping(L, op_file_loc, root_folder):
    model_idx_map = {}
    for i in range(len(L)):
        model_idx_map[i] = L[i][(len(root_folder)+1):]
    with open(op_file_loc, 'w') as f:
        f.write(str(model_idx_map))
    return model_idx_map
    

def calculate_and_store_test_predictions (input_folder, output_folder, name, net, model_arg_dict, X,
 suffix="testscore", match_terms_to_find_model=[], search_at_level=4):
    """Takes a list of model locations and input data(X) and creates following files:
        1. score files named model_<srno>_<name>_<suffix>.h5
        2. A text file containing mapping between model_locations and <srno>
    """
    summ = open(output_folder+os.sep+"prediction_summary.txt",'w')
    map_file_loc = output_folder+os.sep+"model_loc_mapping.txt"
    L = get_model_list_to_run(input_folder,match_terms_to_find_model, search_at_level)
    mp = create_mapping(L, map_file_loc, input_folder)
    for i in range(len(L)):
        assert(L[i][(len(input_folder)+1):]==mp[i])
        if i<10:
            numstr = "0"+str(i)+"_"
        else:
            numstr = str(i)+"_"
        op_score_file_locn = output_folder+os.sep+"model_"+numstr+name+"_"+suffix+".h5"
        # prepare new graph
        tf.reset_default_graph()
        model_graph = net(model_arg_dict)
        predict_and_save(L[i], op_score_file_locn, model_graph, X)
        summ.write("Model: {}\nOutput: {}\n\n".format(L[i],op_score_file_locn)+"\n"+("="*30)+"\n")
    summ.close()


def get_test_data_generator (X_data,Y_data):
    test_serial = get_srno_array([(0, X_data.shape[0]-1)],np.float32)
    X, Y = X_data, Y_data
    return {"test":DataGenerator(X, Y, True), "test_shape":X.shape,
            "test_serial":test_serial}

def predict_and_save(saved_model_path, output_file_loc, model, X):# model here contains model.graph in which weights will be loaded
    model.build_network()
    sess = tf.Session(graph=model.graph)
    Y = np.zeros(shape=(X.shape[0],3))# dummy array
    dic = get_test_data_generator(X,Y)# second array Y is dummy
    t_s = dic["test_shape"][0]# number of samples
    t_gen = dic["test"]
    t_serial = dic["test_serial"]
    model.saver.restore(sess, saved_model_path)
    fn = output_file_loc
    with hf.File(fn,"w") as f:
        grp = f.create_group("test")
        run_prediction_steps(sess,model,t_s,t_gen,t_serial,grp)
    sess.close()
    print("CLOSED")

def run_prediction_steps (sess,model,sz,gen,serial,grp,batch_size=5):
    """Session model size, generator and hdf group """
    grp.create_dataset("serial",dtype=np.float32,data=serial)
    grp.create_dataset("outputs",shape=(sz,3),dtype=np.float32)
    grp.create_dataset("td_outputs",shape=(sz,3),dtype=np.float32)
    run_list = []
    #ORDER: Y_OP Y_TH PROJECTIONS
    run_list.append(model.outputs[0])# probability output vector
    run_list.append(model.outputs[1])# one hot vector output
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
            B,C,D = sess.run(run_list,feed_dict=fd)
            grp["outputs"][i:i+bs] = B
            grp["td_outputs"][i:i+bs] = C
            grp["projections"][i:i+bs] = D
            i = i+bs
        else:
            B,C = sess.run(run_list,feed_dict=fd)
            grp["outputs"][i:i+bs] = B
            grp["td_outputs"][i:i+bs] = C
            i = i+bs
# Helpers for combine_scores_from_h5_file
def _get_h5_file_list_to_combine (root_dir, extra_match_terms=[], search_at_level = 1):
    match_terms = [".h5"]
    for terms in extra_match_terms:
        match_terms.append(terms)
    L = ut.find_paths(root_dir, match_terms, level=search_at_level)
    L.sort()
    print(L)
    return L

def _extract_data_from_prediction_file(file_loc, dtype=np.float32):
    """ returns a tuple of 3 elements:
        (probability_array, onehot_output_array, projections)
    """
    with hf.File(file_loc, "r") as f:
        prob =  f["test/outputs"][:]
        onehot = f["test/td_outputs"][:]
        projxn = f["test/projections"][:]
        prediction_data = (prob.astype(dtype), onehot.astype(dtype), projxn.astype(dtype))
    return prediction_data

class one_hot ():
    def __init__(self):
        with tf.variable_scope("one_hot"):
            self.inp = tf.placeholder(dtype=tf.float32, shape = (None, 3))
            self.amax = tf.argmax(self.inp,1)
            self.op = tf.one_hot(self.amax, tf.shape(self.inp)[1])
        self.sess = tf.Session()
    
    def convert(self, arr):
        """ x.shape=(1,3)
            x = np.array([[0.1,0.2,0.07]], dtype=np.float32)
            (Pdb) ohot.convert(x)
            [array([[ 0.,  1.,  0.]], dtype=float32), array([1])]

        """
        fd = {self.inp:arr}
        return self.sess.run([self.op, self.amax], fd)

def _dump_data_in_h5(prb, ohv, pxn, agmax, op_file_loc, dtype=np.float32):
    with hf.File(op_file_loc, 'w') as f:
        grp = f.create_group('test')
        grp.create_dataset("outputs",dtype=dtype, data=prb)
        grp.create_dataset("td_outputs",dtype=dtype, data=ohv)
        grp.create_dataset("projections",dtype=dtype, data=pxn)
        grp.create_dataset("class_predictions", dtype=dtype, data=agmax)
        
def _create_colums_for_excel(n):
    """
    returns columns for these xls files:

    ensemble_xls = output_folder+os.sep+model_name+"ensemble_"+suffix+".xls"#xl1
    combined_prob_xls = output_folder+os.sep+model_name+"all_prob_"+suffix+".xls"#xl2
    combined_ohv_xls = output_folder+os.sep+model_name+"all_one_hot_"+suffix+".xls"#xl3
    combined_amax_xls = output_folder+os.sep+model_name+"all_argmax_"+suffix+".xls"#xl4
    """
    x = ["ENS_PROB_0","ENS_PROB_1","ENS_PROB_2",
         "ENS_1HOT_0","ENS_1HOT_1","ENS_1HOT_2",
         "PREDICTED_CLASS_ID"
            ]
    
    y = []
    t = []
    for mod in range(n+1):#num of models
        if mod == n:
            t.append("ENSEMBLE")
        else:
            t.append("MODEL_"+str(mod))
        for c in range(3):
            if mod==n:
                col = "COMBINED_CLS_"+str(c)
            else:
                col = "MODEL_"+str(mod)+"_CLS_"+str(c)
            y.append(col)
    z = y
    return (x,y,z,t)


    
    

def combine_predictions_from_h5_file(input_folder, output_folder, model_name, suffix, idx_vol_key_map, match_terms=[], level=1, num_rows=63, dtype=np.float32 ):
    """
    Creates following files in the output_folder:
        1. h5 file with following structure:
    
    Args:
        num_rows : number of rows in each of the arrays in the prediction files.
        dtype : data type of he ensemble prediction values.

    """
    assert(isinstance(idx_vol_key_map, dict))
    fmp = output_folder+os.sep+model_name+"_File_mapping_"+suffix+".txt" # file mapping output location
    ensemble_h5 = output_folder+os.sep+model_name+"ensemble_"+suffix+".h5"
    ensemble_xls = output_folder+os.sep+model_name+"ensemble_"+suffix+".xls"#xl1
    combined_prob_xls = output_folder+os.sep+model_name+"all_prob_"+suffix+".xls"#xl2
    combined_ohv_xls = output_folder+os.sep+model_name+"all_one_hot_"+suffix+".xls"#xl3
    combined_amax_xls = output_folder+os.sep+model_name+"all_argmax_"+suffix+".xls"#xl4
    ensemble_xls_submit = output_folder+os.sep+model_name+"_ensemble_submit"+suffix+".xls"

    ohot = one_hot()
    L = _get_h5_file_list_to_combine(input_folder, match_terms, level)
    mp = create_mapping(L, fmp, input_folder )
    num_models = len(L)

    sum_prb = np.zeros(shape=(num_rows, 3), dtype=dtype)
    sum_ohv = np.zeros(shape=(num_rows, 3), dtype=dtype)
    sum_pxn = np.zeros(shape=(num_rows,95, 69, 1), dtype=dtype)
    ls_prb = []
    ls_ohv = []# lists for conacatenation to an array to be dumbed in excel file
    ls_amax = [] #argmaxs -- actual predicte class code

    for h in range(num_models):
        assert(L[h][(len(input_folder)+1):]==mp[h])
        prb, ohv, pxn = _extract_data_from_prediction_file(L[h], dtype=dtype)
        print(prb.shape, ohv.shape, pxn.shape)
        print(type(prb), type(ohv), type(pxn))
        assert(sum_prb.shape==prb.shape)
        assert(sum_ohv.shape==ohv.shape)
        assert(sum_pxn.shape==pxn.shape)
        sum_prb = sum_prb + prb
        sum_ohv = sum_ohv + ohv
        sum_pxn = sum_pxn + pxn
        amax = np.argmax(ohv, axis=1)
        amax = np.expand_dims(amax, axis=1)

        ls_prb.append(prb)
        ls_ohv.append(ohv)
        ls_amax.append(amax)
    
    final_prb = sum_prb/num_models
    final_pxn = sum_pxn/num_models
    final_ohv, final_amax = ohot.convert(final_prb)
    final_amax = np.expand_dims(final_amax, axis=1)
    final_ens_op = np.concatenate([final_prb, final_ohv, final_amax], axis=1) 

    ls_prb.append(final_prb)
    ls_ohv.append(final_ohv)
    ls_amax.append(final_amax)

    all_prb = np.concatenate(ls_prb, axis=1)
    all_ohv = np.concatenate(ls_ohv, axis=1)
    all_amax = np.concatenate(ls_amax, axis=1)

    xl1, xl2, xl3, xl4 = _create_colums_for_excel(num_models)
    #dumping data
    _dump_data_in_h5(final_prb, final_ohv, final_pxn, final_amax, ensemble_h5)
    df1 = pd.DataFrame(data=final_ens_op, columns=xl1)
    df1.to_excel(ensemble_xls)

    df2 = pd.DataFrame(data=all_prb, columns=xl2)
    df2.to_excel(combined_prob_xls)

    df3 = pd.DataFrame(data=all_ohv, columns=xl3)
    df3.to_excel(combined_ohv_xls)

    df4 = pd.DataFrame(data=all_amax, columns=xl4)
    df4.to_excel(combined_amax_xls)

    df5 = _get_class_label_list_and_volume_id_as_df(final_amax, idx_vol_key_map)
    df5.to_excel(ensemble_xls_submit)


def _get_class_label_list_and_volume_id_as_df(label_id_array, data_index_volume_key_map):
    """Returns a dataframe with three columns VOLUME_ID, LABEL and PREDICTED_CLASS_ID
    VOLUME_ID[i] is taken as data_index_key_map[i], make sure you feed the correct idx-key mapping.
    """
    assert(label_id_array.shape==(108,1))# raising error in case of unintended use(using partial chunks)
    id_to_name = {0:'MSA', 1:'PSP', 2:'PD'}#DO NOT CHANGE
    df_data = []
    for idx in range(label_id_array.shape[0]):
        vid = data_index_volume_key_map[idx]
        pred_cls_id  = label_id_array[idx,0]
        assert(pred_cls_id in [0.0, 1.0, 2.0])
        cls_name = id_to_name[pred_cls_id]
        df_data.append((vid, cls_name, pred_cls_id))
    return pd.DataFrame(data=df_data, columns=['VOLUME_ID', 'LABEL', 'PREDICTED_CLASS_ID'])

if __name__ == "__main__":
    import pdb
    rf = "/home/abhijit/nas_drive/Abhijit/Shubham/ejnmmi-dpnn/Codes/Checkpoints/ROOT_FOR_TESTING"
    L = get_model_list_to_run(rf)
    create_mapping(L,rf+os.sep+"test_map.txt",rf)
    pdb.set_trace()










