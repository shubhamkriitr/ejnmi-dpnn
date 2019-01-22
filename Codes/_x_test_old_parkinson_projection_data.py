#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 17:53:44 2018
For standardizing and pre-processinig tensor-factorized data for pretraining.
@author: shubham
"""

import os
import numpy as np
import h5py as hf
from utility import get_ancestor
import matplotlib.pyplot as plt
import scipy.io as sio
import os


#np.set_printoptions(precision=200)
ROOT_D = get_ancestor(os.getcwd(),2)# some_dir/some_other_dir/current_dir--> fetches some_dir
DSET_FOLDER = ROOT_D+"/Datasets"
DEFAULT_LOC = DSET_FOLDER + os.sep + "pretraining_projection_data_copy.mat"
TF_DATA_LOC = DSET_FOLDER+"/FactorizedResult.mat"
P_DATA_LOC = DSET_FOLDER+"/P_DATA.h5"

def load_mat_file (file_loc):
    return sio.loadmat(file_loc)

def describe_mat (file_loc):
    L = sio.whosmat(file_loc)
    i=0
    for item in L:
        i+=1
        print("Sr.No.",i,item)
    return L

def plot_volume_as_2d (vol, ch_range=None):
    assert(len(vol.shape)==3)
    if ch_range is not None:
        ch = ch_range[1] - ch_range[0] + 1
        c_ch = ch_range[0]
    else:
        ch = vol.shape[2]
        c_ch = 0
    rows = np.sqrt(ch)
    rows = int(rows)
    cols = rows
    if (ch > rows*rows):
        rows = rows +1
    fig, axs = plt.subplots(nrows=rows, ncols=cols)

    for r in range(rows):
        for c in range(cols):
            if c_ch == ch:
                break
            axs[r][c].imshow(vol[:,:,c_ch])
            c_ch+=1
    plt.show()


def create_pretraining_tf_dataset (IP_VOL_LOC, TF_DATA_LOC, OP_LOC):
    """Creates Datset fr pretraining
    Args:
        IP_VOL_LOC: Input MRI vol file location
        TF_DATA_LOC: TF_Data file loc
        OP_LOC: hdf5 file path where outpu will be stored
    """
    summary_path = OP_LOC[:-3]+"_summary.txt"
    summ = open(summary_path,"w")
    summ.write("INPUT VOLUME PATH:"+IP_VOL_LOC)
    summ.write("TF_DATA_LOC:"+TF_DATA_LOC)
    summ.write("OP_LOC:"+OP_LOC)
    f_ip = load_mat_file(IP_VOL_LOC)
    f_tf = load_mat_file(TF_DATA_LOC)

    op = hf.File(OP_LOC,"w")
    op.create_dataset("volumes",shape=(1077,95,69,79,1),dtype=np.float32)
    op.create_dataset("tf_maps",shape=(1077,95,69,1),dtype=np.float32)
    op.create_dataset("labels",shape=(1077,1),dtype=np.float32)

    class_id = 2.0 #using 0,1,2 for PD => 3,4....43 for new classes
    sr_no = -1
    total=0
    for i in range(1,42):
        if i<10:
            z = "Z0"
        else:
            z="Z"
        z = z+str(i)
        total+=f_ip[z].shape[0]
        class_id+=1
        if (len(f_tf[z].shape))>2:
            assert(f_ip[z].shape[0]==f_tf[z].shape[2])
        else:
            assert(f_ip[z].shape[0]==1)
        for j in range(f_ip[z].shape[0]):
            sr_no+=1
            vol = (np.transpose((f_ip[z][j,0]),[0,2,1])).astype(np.float32)
            if (len(f_tf[z].shape))>2:
                tf_map = (f_tf[z][:,:,j]).astype(np.float32)
            else:
                tf_map = (f_tf[z][:,:]).astype(np.float32)
            op["volumes"][sr_no,:,:,:,0] = vol
            op["tf_maps"][sr_no,:,:,0] = tf_map
            op["labels"][sr_no] = class_id

            log = "Sr.No.{} <-- Z={},sample=[{}], label={}, (i,j)=({},{})\n"
            log = log.format(sr_no,z,j,class_id,i,j)
            print(log)
            summ.write(log)

    summ.write("Total="+str(total))

    summ.close()
    op.close()
    
def create_parkinson_tf_dataset (IP_VOL_LOC, TF_DATA_LOC, OP_LOC):
    """Creates Datset containing input volumes and corrsponding tensor-factorized
    maps.
    INFO->Sr.No. 1 ('MSADataList', (95, 69, 91), 'double')
        Sr.No. 2 ('PDDataList', (95, 69, 136), 'double')
        Sr.No. 3 ('PSPDataList', (95, 69, 30), 'double')
    Args:
        IP_VOL_LOC: Input MRI vol file location(use output of _x_parkinson_data.py)
        TF_DATA_LOC: TF_Data file loc
        OP_LOC: hdf5 file path where outpu will be stored
    """
    pd_class_mapping = {0:'MSADataList', 1:'PSPDataList', 2:'PDDataList'}#DO NOT CHANGE
    break_points = [(0,90), (91,120), (121,256)]
    
    summary_path = OP_LOC[:-3]+"_summary.txt"
    summ = open(summary_path,"w")
    summ.write("INPUT VOLUME PATH:"+IP_VOL_LOC)
    summ.write("TF_DATA_LOC:"+TF_DATA_LOC)
    summ.write("OP_LOC:"+OP_LOC)
    summ.write(str(pd_class_mapping)+"\n")
    summ.write("Class Ranges:"+str(break_points)+"\n")
    f_ip = hf.File(IP_VOL_LOC,"r")
    f_tf = load_mat_file(TF_DATA_LOC)

    op = hf.File(OP_LOC,"w")
    op.create_dataset("volumes",shape=(257,95,69,79,1),dtype=np.float32)
    op.create_dataset("tf_maps",shape=(257,95,69,1),dtype=np.float32)
    op.create_dataset("labels",shape=(257,1),dtype=np.float32)
    op.create_dataset("one_hot_labels",shape=(257,3,1),dtype=np.float32)

    class_id = -1.0 #using 0,1,2 for Parkinson Classes
    sr_no = -1
    total=0
    for i in range(3):
        z = pd_class_mapping[i]
        
        class_id+=1
        if (len(f_tf[z].shape))>2:
            n = f_tf[z].shape[2]
            total+=f_tf[z].shape[2]
        else:
            n=1
            total+=1
        assert(n==(break_points[i][1]-break_points[i][0]+1))
        for j in range(n):
            sr_no+=1
            vol = (np.transpose((f_ip["data"][sr_no]),[0,2,1,3])).astype(np.float32)
            if (len(f_tf[z].shape))>2:
                tf_map = (f_tf[z][:,:,j]).astype(np.float32)
            else:
                tf_map = (f_tf[z][:,:]).astype(np.float32)
            op["volumes"][sr_no,:,:,:,:] = vol
            op["tf_maps"][sr_no,:,:,0] = tf_map
            op["labels"][sr_no] = class_id
            op["one_hot_labels"][sr_no,i,0] = 1.0
            assert(f_ip["labels"][sr_no,i,0]==1.0)

            log = "Sr.No.{} <-- Z={},sample=[{}], label={}, (i,j)=({},{})\n"
            log = log.format(sr_no,z,j,class_id,i,j)
            print(log)
            summ.write(log)

    summ.write("Total="+str(total))

    summ.close()
    op.close()
    f_ip.close()




#%%
if __name__ == "__main__":
#      L = describe_mat(TF_DATA_LOC)
#      
#      x = input("=="*20)
      L = describe_mat(DEFAULT_LOC)
#      x = input("=="*20)
#      op_loc = DSET_FOLDER+os.sep+"parkinson_with_tf_data.h5"
#      create_parkinson_tf_dataset(P_DATA_LOC, TF_DATA_LOC, op_loc)
#    op_loc = DSET_FOLDER+os.sep+"pretraining_tensor_factorized_data.h5"
#    create_pretraining_tf_dataset(DEFAULT_LOC, TF_DATA_LOC, op_loc)
    
    

