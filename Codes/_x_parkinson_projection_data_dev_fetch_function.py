#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 10:15:02 2018

@author: reuter
"""
import os
import numpy as np
import h5py as hf
from utility import get_ancestor
CWD = os.getcwd()
ROOT_D = get_ancestor(CWD,2)# some_dir/some_other_dir/current_dir--> fetches some_dir
DSET_FOLDER = ROOT_D+"/Datasets"
DEFAULT_LOC = DSET_FOLDER+"/imdb_NormDat_IXI.mat"

PARKINSON_TF_DATA_LOC = DSET_FOLDER+os.sep+"parkinson_with_tf_data.h5"
def get_parkinson_TF_data(file_loc=None, ranges=[[0,256]]):
    """
    Input:
        file_loc: location of the .hdf5 file.
        ranges = [[a,b],[c,d]...] where samples ranging from [a,b] [c,d] .. to
        be concatenated and returned
        Output:
            tuple of two numpy arrays (Training Inputs, Corresponding Outputs)
    """
    if file_loc is None:
        file_loc = PARKINSON_TF_DATA_LOC
    return _get_parkinson_TF_data(file_loc, ranges)

def _get_parkinson_TF_data(file_loc=None, ranges=[[0, 256]]):
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
        tf_map = f["tf_maps"]
        for j in range(len(ranges)):
            if j!=0:
                X = np.concatenate([X,vol[ranges[j][0]:ranges[j][1]+1]],axis=0)
                Y = np.concatenate([Y,tf_map[ranges[j][0]:ranges[j][1]+1]],axis=0)
            else:
                X = vol[ranges[j][0]:ranges[j][1]+1]
                Y = tf_map[ranges[j][0]:ranges[j][1]+1]
    return (X,Y)

def get_parkinson_classification_data(file_loc=None, ranges=[[0,256]]):
    """
    Input:
        file_loc: location of the .hdf5 file.
        ranges = [[a,b],[c,d]...] where samples ranging from [a,b] [c,d] .. to
        be concatenated and returned
        Output:
            tuple of two numpy arrays (Training Inputs, Corresponding Outputs)
    """
    if file_loc is None:
        file_loc = PARKINSON_TF_DATA_LOC
    return _get_parkinson_classification_data(file_loc, ranges)

def _get_parkinson_classification_data(file_loc=None, ranges=[[0, 256]]):
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
        tf_map = f["one_hot_labels"]
        for j in range(len(ranges)):
            if j!=0:
                X = np.concatenate([X,vol[ranges[j][0]:ranges[j][1]+1]],axis=0)
                Y = np.concatenate([Y,tf_map[ranges[j][0]:ranges[j][1]+1]],axis=0)
            else:
                X = vol[ranges[j][0]:ranges[j][1]+1]
                Y = tf_map[ranges[j][0]:ranges[j][1]+1]
    return (X,Y)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import utility as ut
    print("TF DATA")
    
    test_range = [ [[0,1],[255,256]],[[0,100]],[[0,100],[100,256]]      ]

    for ranges in test_range:
        X,Y=get_parkinson_TF_data(ranges=ranges)
        print("R=",ranges)
        ut.get_array_info(X,"X")
        ut.get_array_info(Y,"Maps")

    wait  = input("wait")
    for ranges in test_range:
        X,Y=get_parkinson_classification_data(ranges=ranges)
        print("R=",ranges)
        ut.get_array_info(X,"X")
        ut.get_array_info(Y,"Y")













