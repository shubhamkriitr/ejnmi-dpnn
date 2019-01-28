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

def _extract_data_from_prediction_file(file_loc):
    """ returns a tuple of 3 elements:
        (probability_array, onehot_output_array, projections)
    """
    with hf.File(file_loc, "r") as f:
        prob =  f["test/outputs"][:]
        onehot = f["test/td_outputs"][:]
        projxn = f["test/projections"][:]
        prediction_data = (prob, onehot, projxn)
    return prediction_data

class one_hot ():
    def __init__(self):
        with tf.variable_scope("one_hot"):
            self.inp = tf.placeholder(dtype=tf.float32, shape = (None, 3))
            self.op = tf.one_hot(tf.argmax(self.input,1), tf.shape(input_)[1])
        self.sess = tf.Session()
    
    def convert(self, arr):
        """arr.shape=(1,3)"""
        fd = {self.inp:arr}
        return self.sess.run([self.op], fd)
        

def combine_predictions_from_h5_file(input_folder, output_folder, model_name, suffix, match_terms=[], level=1, num_rows=63, dtype=np.float32 ):
    """
    Creates following files in the output_folder:
        1. h5 file with following structure:
    
    Args:
        num_rows : number of rows in each of the arrays in the prediction files.
        dtype : data type of he ensemble prediction values.

    """
    fmp = output_folder+os.sep+model_name+"_File_mapping_"+suffix+".txt" # file mapping output location
    L = _get_h5_file_list_to_combine(input_folder, match_terms, level)
    mp = create_mapping(L, fmp, input_folder )
    num_models = len(L)
    sum_prb = np.zeros(shape=(num_rows, 3), dtype=dtype)
    sum_ohv = np.zeros(shape=(num_rows, 3), dtype=dtype)
    sum_pxn = np.zeros(shape=(num_rows,95, 69, 1), dtype=dtype)
    final_ohv = np.zeros(shape=(num_rows, 3), dtype=dtype)
    final_prb = np.zeros(shape=(num_rows, 3), dtype=dtype)
    final_pxn = np.zeros(shape=(num_rows,95, 69, 1), dtype=dtype)
    for h in range(num_models):
        assert(L[h][(len(input_folder)+1):]==mp[h])
        prb, ohv, pxn = _extract_data_from_prediction_file(L[h])
        print(prb.shape, ohv.shape, pxn.shape)
        print(type(prb), type(ohv), type(pxn))



class Accumulator:
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)
        self.arrays = {}
    
    def add_array(self, name, value):
        self._add(name, value)
    
    def _add(self, key, value):
        if key in self.arrays.keys():
            self._check_consistency(key, value)
            self.arrays[key] += value
        else:
            self.arrays[key] = value
    
    def _check_consistency(self, key, value):
        arr = self.arrays[key]
        assert(arr.dtype == value.dtype)
        assert(arr.shape == value.shape)
    
    def _dump_array(self,key,hdf_group):
        arr = self.arrays[key]
        dset = hdf_group.create_create_dataset(key,shape=arr.shape,dtype=np.float32,data=arr)
    
    def _dump_all_arrays(self, hdf_group):
        for key in self.arrays.keys():
            self._dump_array(key, hdf_group)
    
    def dump_all(self, hdf_group):
        self._dump_all_arrays(hdf_group)

if __name__ == "__main__":
    import pdb
    rf = "/home/abhijit/nas_drive/Abhijit/Shubham/ejnmmi-dpnn/Codes/Checkpoints/ROOT_FOR_TESTING"
    L = get_model_list_to_run(rf)
    create_mapping(L,rf+os.sep+"test_map.txt",rf)
    pdb.set_trace()










