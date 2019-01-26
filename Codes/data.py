#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 20:51:15 2017

@author: shubham
"""

# DATA LOADER SCRIPT --Temporary
import os
import numpy as np
import h5py as hf
from utility import get_ancestor
CWD = os.getcwd()
ROOT_D = get_ancestor(CWD,2)# some_dir/some_other_dir/current_dir--> fetches some_dir
DSET_FOLDER = ROOT_D+"/Datasets"
DEFAULT_LOC = DSET_FOLDER+"/imdb_NormDat_IXI.mat"
PARKINSON_TF_DATA_LOC = DSET_FOLDER+os.sep+"parkinson_with_tf_data.h5"
SMOOTH_PARKINSON_TF_DATA_LOC = DSET_FOLDER+os.sep+"smoothed_parkinson_with_tf_data.h5"
SMOOTH_PRETRAINING_TF_DATA_LOC = DSET_FOLDER+os.sep+"smoothed_pretraining_tensor_factorized_data.h5"
## NEW DATASET LOCATIONS
NEW_DSET_FOLDER = ROOT_D+"/Datasets/2019_JAN_EJNMMI"
NEW_PARKINSON_DEV_DATA_LOC_H5 = NEW_DSET_FOLDER + os.sep + "HDF/new_development_data.h5"
NEW_PARKINSON_TEST_DATA_LOC_H5 = NEW_DSET_FOLDER+"/HDF/new_blindtest_data.h5"


def load_mat_file(file_loc):
    return hf.File(file_loc, 'r')


def get_data(file_loc=None, ranges=[[0, 499]]):
    if file_loc is None:
        file_loc = DEFAULT_LOC
    return get_IXI_age_data(file_loc, ranges)


def get_IXI_age_data(file_loc, ranges=[[0, 499]]):
    """Input:
        file_loc: location of the .mat file.
        ranges: A sequence of sequences of two numbers,e.g. [[a,b],[c,d]...]
        where samples ranging from [a,b] [c,d] .. are to be concatenated and
        returned
        Output:
            tuple of two numpy arrays (Training Inputs, Corresponding Outputs)
    """
    X, Y = None, None
    with hf.File(file_loc,'r') as f:
        vol = f["imdb"]["images"]["data"]
        age = f["imdb"]["images"]["Age"]
        for j in range(len(ranges)):
            if j!=0:
                X = np.concatenate([X,vol[ranges[j][0]:ranges[j][1]+1]],axis=0)
                Y = np.concatenate([Y,age[ranges[j][0]:ranges[j][1]+1]],axis=0)
            else:
                X = vol[ranges[j][0]:ranges[j][1]+1]
                Y = age[ranges[j][0]:ranges[j][1]+1]
    if X is not None:
        X = np.expand_dims(X,axis=-1)
    return (X,Y)

def get_parkinson_data(file_loc=DSET_FOLDER+os.sep+"P_DATA.h5",ranges=[[0,256]]):
    #TODO_ add split_ratio, and chunk_number as arguments
    pd_class_mapping = {0:'MSADataList', 1:'PSPDataList', 2:'PDDataList'}#DO NOT CHANGE
    break_points = [(0,90), (91,120), (121,256)]
    return _get_parkinson_data(file_loc, ranges)

def _get_parkinson_data(file_loc=None, ranges=[[0, 256]]):
    """
    Input:
        file_loc: location of the .hdf5 file. If None, then the location
        ../Datasets/P_DATA.h5 is used.
        ranges = [[a,b],[c,d]...] where samples ranging from [a,b] [c,d] .. to
        be concatenated and returned
        Output:
            tuple of two numpy arrays (Training Inputs, Corresponding Outputs)
    """
    X, Y = None, None
    with hf.File(file_loc,'r') as f:
        vol = f["data"]
        class_id = f["labels"]
        for j in range(len(ranges)):
            if j!=0:
                X = np.concatenate([X,vol[ranges[j][0]:ranges[j][1]+1]],axis=0)
                Y = np.concatenate([Y,class_id[ranges[j][0]:ranges[j][1]+1]],axis=0)
            else:
                X = vol[ranges[j][0]:ranges[j][1]+1]
                Y = class_id[ranges[j][0]:ranges[j][1]+1]
    return (X,Y)

def get_pretraining_TF_data(file_loc=None, ranges=[[0,1076]]):
    """
    Input:
        file_loc: location of the .hdf5 file.
        ranges = [[a,b],[c,d]...] where samples ranging from [a,b] [c,d] .. to
        be concatenated and returned
        Output:
            tuple of two numpy arrays (Training Inputs, Corresponding Outputs)
    """
    if file_loc is None:
        file_loc = DSET_FOLDER+os.sep+"pretraining_tensor_factorized_data.h5"
    return _get_pretraining_TF_data(file_loc, ranges)

def get_smoothed_pretraining_TF_data(file_loc=None, ranges=[[0,1076]]):
    """
    Input:
        file_loc: location of the .hdf5 file.
        ranges = [[a,b],[c,d]...] where samples ranging from [a,b] [c,d] .. to
        be concatenated and returned
        Output:
            tuple of two numpy arrays (Training Inputs, Corresponding Outputs)
    """
    if file_loc is None:
        file_loc = SMOOTH_PRETRAINING_TF_DATA_LOC
    return _get_pretraining_TF_data(file_loc, ranges)

def _get_pretraining_TF_data(file_loc=None, ranges=[[0, 256]]):
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

def get_smoothed_parkinson_TF_data(file_loc=None, ranges=[[0,256]]):
    """
    Input:
        file_loc: location of the .hdf5 file.
        ranges = [[a,b],[c,d]...] where samples ranging from [a,b] [c,d] .. to
        be concatenated and returned
        Output:
            tuple of two numpy arrays (Training Inputs, Corresponding Outputs)
    """
    if file_loc is None:
        file_loc = SMOOTH_PARKINSON_TF_DATA_LOC
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

def get_smoothed_parkinson_classification_data(file_loc=None, ranges=[[0,256]]):
    """
    Input:
        file_loc: location of the .hdf5 file.
        ranges = [[a,b],[c,d]...] where samples ranging from [a,b] [c,d] .. to
        be concatenated and returned
        Output:
            tuple of two numpy arrays (Training Inputs, Corresponding Outputs)
    """
    if file_loc is None:
        file_loc = SMOOTH_PARKINSON_TF_DATA_LOC
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

def carve_out_chunk(A,B, ranges=[[0, 256]]):
    """Returns a tuple of numpy arrays having elements selected from A and B
    respectively from the ranges specified in `ranges`.
    Input:
        A: numpy array(input dataset)
        B: numpy array(output dataset)
        ranges = [[a,b],[c,d]...] where samples ranging from [a,b] [c,d] .. to
        be concatenated and returned
        Output:
            tuple of two numpy arrays (Training Inputs, Corresponding Outputs)
    """
    X, Y = None, None
    for j in range(len(ranges)):
        if j!=0:
            X = np.concatenate([X,A[ranges[j][0]:ranges[j][1]+1]],axis=0)
            Y = np.concatenate([Y,B[ranges[j][0]:ranges[j][1]+1]],axis=0)
        else:
            X = A[ranges[j][0]:ranges[j][1]+1]
            Y = B[ranges[j][0]:ranges[j][1]+1]
    return (X,Y)

#%% New Parkinson Dataset
def get_new_dev_parkinson_cls_data(file_loc=None, ranges=[[0,245]]):
    """
    Input:
        file_loc: location of the .hdf5 file.
        ranges = [[a,b],[c,d]...] where samples ranging from [a,b] [c,d] .. to
        be concatenated and returned
        Output:
            tuple of two numpy arrays (Training Inputs, Corresponding Outputs)
    """
    if file_loc is None:
        file_loc = NEW_PARKINSON_DEV_DATA_LOC_H5
    return _get_new_dev_parkinson_cls_data(file_loc, ranges)

def _get_new_dev_parkinson_cls_data(file_loc=None, ranges=[[0, 245]]):
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

## New Test Dataset
def get_new_test_parkinson_cls_data(file_loc=None, ranges=[[0,62]]):
    """
    Input:
        file_loc: location of the .hdf5 file.
        ranges = [[a,b],[c,d]...] where samples ranging from [a,b] [c,d] .. to
        be concatenated and returned
        Output:
            tuple of two numpy arrays (Training Inputs, Corresponding Outputs)
    """
    if file_loc is None:
        file_loc = NEW_PARKINSON_TEST_DATA_LOC_H5
    return _get_new_test_parkinson_cls_data(file_loc, ranges)

def _get_new_test_parkinson_cls_data(file_loc=None, ranges=[[0, 62]]):
    """
    Input:
        file_loc: location of the .hdf5 file.
        ranges = [[a,b],[c,d]...] where samples ranging from [a,b] [c,d] .. to
        be concatenated and returned
        Output:
            tuple containing a numpy array (Training Inputs,)
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

#%%

class DataGenerator ():
    def __init__(self, x, y, verbose=False):
        self.X = None#input Dataset(s)
        self.Y = None#output/label dataset(s)
        self.last = -1  # last index of the random index array which was used
                         # to retrieve the data from the datasets
        self.d_size = None  # size of the dataset
        self.ind = None # array of random indices used for retrieving random
                        # samples from the dataset.
        self.c_ind = None # array of continuous indices used for retrieving contiguous
                          #samples from the dataset
        self.verbose = verbose
        self._shuffle_before_use = False # used to indicate whether or not to
                                         # shuffle the ind array
        self._set_X_and_Y(x,y)
        self.t_last = -1  # last index from which the sample has been taken out
                          # of the dataset for "testing/evaluation". Used by the
                          # function get_next_test_batch, which fetches contiguous
                          # chunks of data(No random shuffling).

    def get_dataset_size(self):
        assert(self.d_size is not None)
        return self.d_size

    def _set_X_and_Y(self, X, Y):
        #TODO_ need to be altered for MIMO models, in that case X & Y will
        # be lists of ndarrays.
        if (X is None or Y is None):
            return
        self.X = X
        self.Y = Y
        assert(self.X.shape[0] == self.Y.shape[0])
        self.last = -1
        self.d_size = self.X.shape[0]
        self.ind = np.linspace(0,self.d_size-1,self.d_size,
                                       dtype=np.int64)
        np.random.shuffle(self.ind)
        self.c_ind = np.linspace(0,self.d_size-1,self.d_size,
                                       dtype=np.int64)
        self.print("Initial Random Indices", self.ind)
        self.print("Initial Continuous Indices",self.c_ind)

    def _fetch(self, ranges):
        """Extracts shuffled samples from the datasets using values of self.ind
        at indices specififed in the ranges.

            Args:
                ranges(:obj: `list` of :obj: `list`): A sequence of sequences
                of two numbers. e.g. [[a,b], [c, d]]. Where a <= b, c <= d..
                and both elements are inclusive.
        """

        if len(ranges) == 1:
            idx = self.ind[ranges[0][0]:ranges[0][1]+1]
            return (self.X[idx],
                    self.Y[idx]
                    )
        else:
            fetch_x = []
            fetch_y = []
            for r in ranges:
                idx = self.ind[r[0]:r[1]+1]
                fetch_x.append(self.X[idx])
                fetch_y.append(self.Y[idx])
            fetch_x = np.concatenate(fetch_x, axis=0)
            fetch_y = np.concatenate(fetch_y, axis=0)
            return (fetch_x, fetch_y)

    def fetch_in_order(self, ranges):
        """Extracts samples from the datasets specififed in the ranges.

            Args:
                ranges: A sequence of sequences of two numbers. e.g. [[a,b],
                [c, d]]. Where a <= b, c <= d.. both elements are inclusive.

        """

        if len(ranges) == 1:
            idx = self.c_ind[ranges[0][0]:ranges[0][1]+1]
            return (self.X[idx],
                    self.Y[idx]
                    )
        else:
            fetch_x = []
            fetch_y = []
            for r in ranges:
                idx = self.c_ind[r[0]:r[1]+1]
                fetch_x.append(self.X[idx])
                fetch_y.append(self.Y[idx])
            fetch_x = np.concatenate(fetch_x, axis=0)
            fetch_y = np.concatenate(fetch_y, axis=0)
            return (fetch_x, fetch_y)

    def _get_next_range(self, batch_size):
        assert(batch_size <= self.d_size)
        if (self._shuffle_before_use):
            np.random.shuffle(self.ind)
            self.print("Indices shuffled:", self.ind)
            self._shuffle_before_use = False
        if (self.last + batch_size) <= (self.d_size-1):
            r = [[self.last + 1, self.last + batch_size]]
            self.last = self.last + batch_size
        else:
            if self.last == (self.d_size - 1):
                r = [[0, batch_size - 1]]
                self.last = (batch_size - 1)
                np.random.shuffle(self.ind)
                self.print("Indices shuffled:", self.ind)
            else:
                r = [[self.last + 1, self.d_size-1]]
                v = batch_size - (self.d_size - self.last)
                r.append([0, v])
                self.last = v
                self._shuffle_before_use = True
        return r

    def get_next_batch(self, batch_size):
        """Fetches a random batch of data of the gievn batch_size. """
        ranges = self._get_next_range(batch_size=batch_size)
        return self._fetch(ranges=ranges)


    def print(self, *args):
        if self.verbose:
            print(*args)

    def reset_state_of_test_generator(self):
        self.t_last = -1

    def get_next_test_batch(self, batch_size):
        """Fetches data in order(No Shuffling). Intended to be used during eva-
        luation of the models. N.B.: Make sure to call
        reset_state_of_test_generator before starting
        to fetch data using this function.
        """
        if (self.t_last + batch_size) < (self.d_size-1):
            x, y = self.fetch_in_order([[self.t_last+1, self.t_last+batch_size]])
            self.t_last+=batch_size
            return (x, y, batch_size)
        elif self.t_last<(self.d_size-1):
            batch_size = (self.d_size-1)-self.t_last
            x, y = self.fetch_in_order([[self.t_last+1, self.t_last+batch_size]])
            self.t_last+=batch_size
            return (x, y, batch_size)
        else:
            return (None, None, None)


if __name__ == '__main__':
    pass
#%%
#    print(PD)
#    file_loc = DSET_FOLDER+"/imdb_NormDat_IXI.mat"#"/imdb.mat"
#    x = load_mat_file(file_loc)
#    # print(x)
#    i = 0
#    for keys in x.keys():
#        i+=1
#        print(i,":",keys)
#    i = 0
#    for keys in x["#refs#"].keys():
#        i+=1
#        print(i,keys,type(x["#refs#"][keys]),x["#refs#"][keys].shape)
#    i = 0
#    for keys in x["imdb"].keys():
#        i+=1
#        print(i,keys,type(x["imdb"][keys]))
#    i = 0
#    for keys in x["imdb"]["images"].keys():
#        i+=1
#        print(i,keys,type(x["imdb"]["images"][keys]),x["imdb"]["images"][keys].shape,
#                          x["imdb"]["images"][keys].dtype)
#
#    X, Y = get_data(ranges=[[0,0],[587,587]])
#    print ("X:",X.dtype,X.shape)
#    print ("Y:",Y.dtype,Y.shape)
#%%
#    X = np.linspace(100, 126, 26, dtype=np.int64)
#    Y = np.linspace(200, 226, 26, dtype=np.int64)
#%%
#    X = np.linspace(100, 108, 8, dtype=np.int64)
#    Y = np.linspace(200, 108, 8, dtype=np.int64)
#    print(X,Y)
#    dgen = DataGenerator(X, Y, True)
#    dsz = 3
#    for i in range(10):
#        print("i=", i)
#        if (i==7):
#            dsz=8
#        x = dgen._get_next_range(dsz)
#        print(x)
#        print("-"*25)
#    print("="*100)
#    del dgen
#    dgen = DataGenerator(X, Y, True)
#    dsz = 4
#    for i in range(10):
#        print("i=", i)
#        if (i==6):
#            dsz=8
#        x = dgen._get_next_range(dsz)
#        print(x)
#        print("-"*25)
#
#    del dgen
#    X = np.linspace(100, 110, 10, dtype=np.int64)
#    Y = np.linspace(200, 110, 10, dtype=np.int64)
#    dgen = DataGenerator(X, Y, True)
#    dsz = 3
#    for i in range(10):
#        print("i=", i)
#        x = dgen._get_next_range(dsz)
#        print(x)
#        print("-"*25)
#
#    del dgen
#    X = np.linspace(100, 108, 8, dtype=np.int64)
#    Y = np.linspace(200, 108, 8, dtype=np.int64)
#    dgen = DataGenerator(X, Y, True)
#    dsz = 2
#    for i in range(10):
#        print("i=", i)
#        x = dgen._get_next_range(dsz)
#        print(x)
#        print("-"*25)
#%%

#    X, Y = get_parkinson_data(ranges=[[0,90],[91,120],[121,256]])
#    del X
#    print(np.sum(Y[0:90,:,:],axis=0))
#    print(np.sum(Y[91:121,:,:],axis=0))
#    print(np.sum(Y[121:257,:,:],axis=0) )


##    X, Y = get_parkinson_data(ranges=[[0,2],[91,93],[121,123]])
#    with hf.File("X.h5","w") as f:
#        f.create_dataset("data",shape=(9,95,79,69,1),dtype=np.float32,data=X)
#        f.create_dataset("labels",shape=(9,3,1),dtype=np.float32,data=Y)
#%% TEST TO SEE THE DATA FETCHED IS A COPY OF DATASET SAMPLES OR A REFERENCE
    print("TEST TO SEE THE DATA FETCHED IS A COPY",
          "OF DATASET SAMPLES OR REFERENCE")
    N = 20
    X = np.array(list(range(0,N)),dtype=np.float32).reshape(N,1)
    Y = np.array(list(range(0,N)),dtype=np.float32).reshape(N,1)
    dgen = DataGenerator(X,Y,True)
    x,y,_  = dgen.get_next_test_batch(5)
    x[0,0] = 100
    y[0,0] = 1000
    print("x",x,x is X[0:1])
    print("y",y)
    print("X",X)
    print("Y",Y)
    x,y  = dgen.get_next_batch(5)
    x[0,0] = 500
    y[0,0] = 5000
    print("x",x)
    print("y",y)
    print("X",X)
    print("Y",Y)