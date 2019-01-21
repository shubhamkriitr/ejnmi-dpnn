#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 16:33:13 2017

@author: shubham
"""

from _visualize import VolumeViewer as VV
import data as dt
import numpy as np
from scipy.ndimage import shift
from scipy.stats import threshold
import utility as ut
X, _ = dt.get_data(ranges=[[15, 15]])
# Y, _ = dt.get_data(ranges=[[15, 15]])
padding = 60
nsz = 256 + 2*padding
def get_mesh (block_size=10,num_blocks=[10,10,10]):
    shape = (block_size, block_size, block_size)
    wht = np.ones(shape=shape,dtype=np.float32)
    blk = np.zeros(shape=shape,dtype=np.float32)
    col_even = []
    col_odd = []
    for i in range(num_blocks[0]):
        if i%2==0:
            col_even.append(wht)
            col_odd.append(blk)
        else:
            col_even.append(blk)
            col_odd.append(wht)
    col_even = np.concatenate(col_even,axis=0)
    col_odd = np.concatenate(col_odd,axis=0)
    layer = []
    layer_odd = []
    for i in range(num_blocks[1]):
        if i%2==0:
            layer.append(col_even)
            layer_odd.append(col_odd)
        else:
            layer.append(col_odd)
            layer_odd.append(col_even)
    layer = np.concatenate(layer,axis=1)
    layer_odd = np.concatenate(layer_odd,axis=1)
    vol = []
    for i in range (num_blocks[2]):
        if i%2==0:
            vol.append(layer)
        else:
            vol.append(layer_odd)
    vol = np.concatenate(vol,axis=2)
    return vol

def reduce_dim (arr, channel=0):
    """channel is the channel to be preserved"""
    return arr[...,channel]

def translate_batch(X_in, dr_max):
    """Function to translate the volumes given in batch X by random
    number of pixels.

    Args:
        X_in: A batch of 3D arrays. Shape of X must be (batch_size,L,W,H,1)
        dr_max: a sequence of 3 non-negative integers e.g [5, 7, 3]

    Returns:
        An array of same shape as that of X, where each volume has been trans-
        lated by pixels p in the range [-dr_max[i],dr_max[i]] along ith axis.
        i in [0, 1, 2]
    """
    X = np.copy(X_in)
    assert (len(X.shape) == 5 and X.shape[-1] == 1)
    for i in range(X.shape[0]):
        X[i, :, :, :, 0] = np.clip(_translate_volume(X[i, :, :, :, 0], dr_max),
                                   0.00001, 1.0)

    return X


def _translate_volume(vol, dr):
    dR = [0, 0, 0]
    for i in range(3):
        dR[i] = np.random.randint(-abs(dr[i]), abs(dr[i]) + 1)
    print ("dR:",dR)# DEV_
    return shift(vol, dr)


if __name__ == "__main__":
#    L = translate_batch(X,[20,20,20])
#    vv = VV(X)
#    ww = VV(L)
    f, q =np.unique(X,return_counts=True)
    print ("Counts:",f)
    print ("Nums:",q)




