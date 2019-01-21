#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 12:30:48 2018

@author: shubham
"""

import os
import numpy as np
import h5py as hf
from utility import get_ancestor
import scipy.io as sio
#import os



CWD = os.getcwd()
ROOT_D = get_ancestor(CWD,2)# some_dir/some_other_dir/current_dir--> fetches some_dir
DSET_FOLDER = ROOT_D+"/Datasets"
TF_DATA_LOC = DSET_FOLDER+"/FactorizedResult.mat"


DEFAULT_LOC = DSET_FOLDER + os.sep + "pretraining_projection_data_copy.mat"

def load_mat_file (file_loc):
    return sio.loadmat(file_loc)


def analyze (file_path=None):
    if file_path is None:
        file_path = DEFAULT_LOC
    try:
        with hf.File(file_path,'r') as f:
            print(f.keys())
    except OSError:
        print("H5py unable to read.")
        f = load_mat_file(file_path)
        print(f.keys())

analyze(TF_DATA_LOC)