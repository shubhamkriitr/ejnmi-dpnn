#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed 20 Feb 2019 19:36 

HDF file creation for new extended dataset.
@author: shubham
"""

import os
import numpy as np
import h5py as hf
from utility import get_ancestor
import matplotlib.pyplot as plt
import scipy.io as sio



ROOT_D = get_ancestor(os.getcwd(), 2)
NEWEXT_DSET_FOLDER = ROOT_D+"/Datasets/2019_FEB_EJNMMI"
NEWEXT_DEV_DATA_LOC = NEWEXT_DSET_FOLDER+ "/train/" + "NewTrainingData.mat"
NEWEXT_TEST_DATA_LOC = NEWEXT_DSET_FOLDER+ "/test/" + "NewBlindTestData.mat"
NEWEXT_DEV_DATA_LOC_H5 = NEWEXT_DSET_FOLDER + "/HDF_EXT/" +  "new_extended_development_data.h5"
NEWEXT_TEST_DATA_LOC_H5 = NEWEXT_DSET_FOLDER+ "/HDF_EXT/" +  "new_extended_blindtest_data.h5"

def load_mat_file (file_loc):
    return sio.loadmat(file_loc)

# Code for creating New Extended TEST Dataset 
def get_sorted_array_of_keys(data_dict, condition = None):
    """Generates a sorted array of keys from dataset

    Creates a sorted list of keys corresponding to array items,
    ignoring the keys for non-array items

    Args:
        data_dict : `dictionary` returned by `load_mat_file`
        condition : A function taking key(`string`) as argument and returning
        True if that key refers to an array item and False otherwise.
    """
    sl = []
    # print(data_dict)
    print(data_dict.keys())
    for key in data_dict.keys():
        if condition(key):
            sl.append(key)
    sl.sort()
    return tuple(sl)

def condition_on_test_data_keys(key):
    if ("sw" in key):
        return True
    return False

def create_new_test_dataset(ip_file_loc, op_file_loc):
    m = load_mat_file(ip_file_loc) # data_dict
    k = get_sorted_array_of_keys(m, condition_on_test_data_keys)

    summary_path = op_file_loc[:-3]+"_summary.txt"
    summ = open(summary_path,"w")
    summ.write("\nInput file location: "+ip_file_loc)
    summ.write("\nOutput file location: "+op_file_loc)
    
    assert(len(k)==108) # Num of volumes

    shape = (108,95,69,79,1)
    summ.write("\nOutput Dataset shape: "+str(shape))
    op = hf.File(op_file_loc,"w")
    op.create_dataset("volumes",shape=(108,95,69,79,1),dtype=np.float32)

    for i in range(len(k)):
        key = k[i]
        assert(m[key].shape == (95,79,69))
        d = (m[key]).reshape(95,79,69,1).astype(np.float32)
        d = np.transpose(d, [0,2,1,3])
        assert(d.shape==(95,69,79,1))
        op["volumes"][i,:,:,:,:] = d
        summ.write("\nSr. No. "+str(i)+" <- "+key)
    
    op.close()
    summ.close()




# Code for creating New Extended Development Dataset 
##

def  condition_on_dev_data_keys(key):
    for k in ["MSA", "PSP", "PD"]:
        if k in key:
            return True
    return False

def _class_from_key(key):
    for k in ["MSA", "PSP", "PD"]:
        if k in key:
            return k
    raise(AssertionError('Key is similar to none of the allowed class.'))


def create_new_dev_dataset(ip_file_loc, op_file_loc):
    m = load_mat_file(ip_file_loc) # data_dict
    k = get_sorted_array_of_keys(m, condition_on_dev_data_keys)
    label_to_name = {0:'MSA', 1:'PSP', 2:'PD'}#DO NOT CHANGE
    name_to_label = {'MSA':0, 'PSP':1, 'PD':2}
    # chunks: when the keys(MSA_<n>,..,PD_<m>...PSP_.. ) are arranged in ascending order
    # the keys correponding to MSA(label=0), PSP(label=1) and PD(label=2)
    # lie in the following set of closed ranges.
    chunks = {'MSA':(0,90), 'PSP':(240,289), 'PD':(91,239)}
    #which means break_points will be ..
    break_points = [(0,90), (91,140), (141,289)]

    summary_path = op_file_loc[:-3]+"_summary.txt"
    summ = open(summary_path,"w")
    summ.write("\nLabel to name: "+str(label_to_name))
    summ.write("\nName to label: "+str(name_to_label))
    summ.write("\nChunks : "+str(chunks))
    summ.write("Break points"+str(break_points))
    summ.write("\nInput file location: "+ip_file_loc)
    summ.write("\nOutput file location: "+op_file_loc)
    
    N = 290
    assert(len(k)==N) # Num of volumes

    shape = (N,95,69,79,1)
    summ.write("\nOutput Dataset['volumes'] shape: "+str(shape))
    summ.write("\nOutput Dataset['labels'] shape: "+str((N,1)))
    summ.write("\nOutput Dataset['one_hot_labels'] shape: "+str((N,3,1)))
    op = hf.File(op_file_loc,"w")
    op.create_dataset("volumes",shape=shape,dtype=np.float32)
    op.create_dataset("labels",shape=(N,1),dtype=np.float32)
    op.create_dataset("one_hot_labels",shape=(N,3,1),dtype=np.float32)

    i = -1 # current index on output dataset
    for class_id in range(3):
        class_name = label_to_name[class_id]
        summ.write("\n"+"-"*10+"c={} -- {}".format(class_id, class_name)+"-"*10)
        r = chunks[class_name]
        summ.write("\nWill use chunk = "+str(r))
        summ.write("\n" + "**"*10)
        assert(break_points[class_id][0]==(i+1))
        summ.write("\nStarting at index:"+str(i+1))
        for u in range(r[0],r[1]+1):# u is current index on key list `k`
            i+=1
            key = k[u]
            assert(_class_from_key(key)==class_name)
            assert(m[key].shape == (95,79,69))
            d = (m[key]).reshape(95,79,69,1).astype(np.float32)
            d = np.transpose(d, [0,2,1,3])
            assert(d.shape==(95,69,79,1))
            op["volumes"][i,:,:,:,:] = d
            op["labels"][i,0] = class_id
            op["one_hot_labels"][i,class_id,0] = 1.0
            logstr = "\n"+"OP_idx="+str(i)+" key:"+key+" key_idx:"+str(u)
            logstr += " ClassId:"+str(class_id)+" Name:"+class_name
            summ.write(logstr)

        assert(break_points[class_id][1]==i)
        summ.write("\nEnding at index="+str(i))
        summ.write("\n"+"**"*10)
    op.close()
    summ.close()


#%% Fetch functions for new_ext dataset
#%% New ExtParkinson Dataset
def get_newext_dev_parkinson_cls_data(file_loc=None, ranges=[[0,289]]):
    """
    Input:
        file_loc: location of the .hdf5 file.
        ranges = [[a,b],[c,d]...] where samples ranging from [a,b] [c,d] .. to
        be concatenated and returned
        Output:
            tuple of two numpy arrays (Training Inputs, Corresponding Outputs)
    """
    if file_loc is None:
        file_loc = NEWEXT_DEV_DATA_LOC_H5
    return _get_newext_dev_parkinson_cls_data(file_loc, ranges)

def _get_newext_dev_parkinson_cls_data(file_loc=None, ranges=[[0, 289]]):
    """
    Input:
        file_loc: location of the .hdf5 file.
        ranges = [[a,b],[c,d]...] where samples ranging from [a,b] [c,d] .. to
        be concatenated and returned
        Output:
            tuple of two numpy arrays (Training Inputs, Corresponding Outputs)
    """
    X, Y = None, None
    with hf.File(file_loc,'r') as f:
        vol = f["volumes"]
        oh_label = f["one_hot_labels"]
        for j in range(len(ranges)):
            if j!=0:
                X = np.concatenate([X,vol[ranges[j][0]:ranges[j][1]+1]],axis=0)
                Y = np.concatenate([Y,oh_label[ranges[j][0]:ranges[j][1]+1]],axis=0)
            else:
                X = vol[ranges[j][0]:ranges[j][1]+1]
                Y = oh_label[ranges[j][0]:ranges[j][1]+1]
    return (X,Y)

## New EXT Test Dataset
def get_newext_test_parkinson_cls_data(file_loc=None, ranges=[[0,107]]):
    """
    Input:
        file_loc: location of the .hdf5 file.
        ranges = [[a,b],[c,d]...] where samples ranging from [a,b] [c,d] .. to
        be concatenated and returned
        Output:
            tuple of a numpy array (Test Inputs, )
    """
    if file_loc is None:
        file_loc = NEWEXT_TEST_DATA_LOC_H5
    return _get_newext_test_parkinson_cls_data(file_loc, ranges)

def _get_newext_test_parkinson_cls_data(file_loc=None, ranges=[[0, 107]]):
    """
    Input:
        file_loc: location of the .hdf5 file.
        ranges = [[a,b],[c,d]...] where samples ranging from [a,b] [c,d] .. to
        be concatenated and returned
        Output:
            tuple of a numpy array (Test Inputs, )
    """
    X = None
    with hf.File(file_loc,'r') as f:
        vol = f["volumes"]
        for j in range(len(ranges)):
            if j!=0:
                X = np.concatenate([X,vol[ranges[j][0]:ranges[j][1]+1]],axis=0)
            else:
                X = vol[ranges[j][0]:ranges[j][1]+1]
    return (X,)


if __name__ == '__main__':
    create_new_test_dataset(NEWEXT_TEST_DATA_LOC, NEWEXT_TEST_DATA_LOC_H5)
    create_new_dev_dataset(NEWEXT_DEV_DATA_LOC, NEWEXT_DEV_DATA_LOC_H5)
    X, Y = get_newext_dev_parkinson_cls_data()
    Z, = get_newext_test_parkinson_cls_data()
    import utility as ut
    ut.get_array_info(X, "New Training Data")
    ut.get_array_info(Y, "Y")
    ut.get_array_info(Z, "New Test Data")
    max_v =  2.54515
    min_v = -0.0328023