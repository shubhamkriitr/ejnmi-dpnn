#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 22:04:40 2018

@author: shubham
"""

import os
import numpy as np
import h5py as hf
from utility import get_ancestor
np.set_printoptions(precision=200)
import scipy.io as sio
import os

def load_mat_file (file_loc):
    return sio.loadmat(file_loc)

CWD = os.getcwd()
ROOT_D = get_ancestor(CWD,2)# some_dir/some_other_dir/current_dir--> fetches some_dir
DSET_FOLDER = ROOT_D+"/Datasets"
F_LOC = DSET_FOLDER+"/p_data.mat"

#dict_keys(['__version__', '__globals__', 'PSPDataList', '__header__', 'PDDataList', 'MSADataList'])

data_keys = ['MSADataList', 'PSPDataList', 'PDDataList']
pd_class_mapping = {0:'MSADataList', 1:'PSPDataList', 2:'PDDataList'}




def store_parkinson_data_in_hdf_as_it_is (ofn="P_DATA_HF.h5"):
    f = load_mat_file(F_LOC)
    print(f.keys())
    with hf.File(ofn,'w') as g:
        for key in f.keys():
            if key in data_keys:
                print(key,f[key].shape,f[key][0][0].shape)
                shp = [f[key].shape[0]]
                for dim in f[key][0][0].shape:
                    shp.append(dim)
                shp.append(1)
                shp = tuple(shp)
                print("DSET_SHAPE:", shp)
                g.create_dataset(key,shp,dtype=np.float32)
                sample_shp = list(f[key][0][0].shape)
                sample_shp.append(1)
                sample_shp = tuple(sample_shp)
                for i in range(shp[0]):
                    print("Sample Num:",i,"being processed.")
                    g[key][i] = ((f[key][i][0]).reshape(sample_shp)).astype(dtype=np.float32)

def store_parkinson_data_with_labels (ofn = "P_DATA.h5",F_LOC=F_LOC):
    f = load_mat_file(F_LOC)
    g = hf.File(ofn,"w")
    print(f.keys())
    class_id = -1
    i = -1
    g.create_dataset( "data",shape=(257,95, 79, 69,1), dtype=np.float32)
    g.create_dataset( "labels", shape = (257,3,1), dtype=np.float32, data = np.zeros((257,3,1),np.float32))
    for key in data_keys:
        print(key,f[key].shape,f[key][0][0].shape,f[key][0][0].shape, (f[key][0][0]).dtype)
        n = (f[key].shape)[0]
        class_id += 1
        print("n=",n,"class_id: ",class_id)
        for c_i in range(n):
            i+=1
            print(i,"<--",c_i,end=" ")
            g["data"][i] = ((f[key][c_i][0]).reshape(95, 79, 69, 1)).astype(dtype=np.float32)
            g["labels"][i,class_id,0] = 1.0
            print(g["labels"][i,:,0])
    g.close()


def summarize_parkinson_dataset (F_LOC):
    f = load_mat_file(F_LOC)
    for key in f.keys():
        print(key)
    for key in f.keys():
        if key in data_keys:
            print("-"*20)
            print(key,f[key].shape,f[key][0][0].shape)
        else:
            print(key, f[key])
            print("-"*20)

if __name__ == '__main__':
    summarize_parkinson_dataset(F_LOC)
