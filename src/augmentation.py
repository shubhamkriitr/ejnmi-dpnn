#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 22:52:46 2017

@author: shubham
"""
import numpy as np
from scipy.ndimage import shift


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
    return shift(vol, dr)

def _add_gaussian_noise (X,factor, mean,std_dev,clip_min,clip_max):
    print ("Args:",factor,mean,std_dev,clip_min,clip_max) # DEV_
    shape = X.shape
    gauss = np.random.normal(mean,std_dev,shape)
    gauss = gauss.reshape(1,shape[1],shape[2],shape[3],1)
    noisy_X = X + gauss*factor
    print (np.unique(noisy_X,return_counts=True))
    noisy_X = np.clip(noisy_X,clip_min,clip_max)
    return noisy_X

def add_gaussian_noise (X, factor, mean, std_dev, min_pix_value=0.0,
                               max_pix_value=1.0):
    """Adds gaussian noise to a the given batch.

    Args:
        X: A batch of 3D arrays. Shape of X must be (batch_size,L,W,H,1)
        factor: A list of two floats [f_min,f_max]
        mean: A list of two floats [u_min,u_max]
        std_dev: A list of two floats [s_min,s_max]
        min_pix_value: min allowed pixel value in output array
        max_pix_value: max alloed pixel value in output array

    Returns:
        An array (X+N) after clipping its pixel values to be in the range
        [min_pix_value, max_pix_value]. Where N is a noise tensor whose
        elements have been drawn from a gaussian distribution with mean = u,
        peak_value = f and std. deviation = s. u,f and s are randomly selected
        from the range specified by the arguments mean, factor and std_dev res-
        pectively.
    """
    factor = np.random.uniform(factor[0],factor[1])
    mean = np.random.uniform(mean[0], mean[1])
    std_dev = np.random.uniform(std_dev[0], std_dev[1])
    return _add_gaussian_noise(X,factor, mean,std_dev, min_pix_value,
                               max_pix_value)


