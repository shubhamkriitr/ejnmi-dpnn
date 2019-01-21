#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 16:33:13 2017

@author: shubham
"""

from _visualize import VolumeViewer as VV
import data as dt
import numpy as np
from scipy.ndimage import affine_transform, map_coordinates, shift
from scipy.stats import threshold
import utility as ut
X, _ = dt.get_data(ranges=[[15,15]])

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

def transform_matrix_offset_center(matrix, x, y, z):
  o_x = float(x) / 2 + 0.5
  o_y = float(y) / 2 + 0.5
  o_z = float(z) / 2 + 0.5

  offset_matrix = np.array([[1, 0, 0, o_x], [0, 1, 0, o_y],
                            [0, 0, 1, o_z],[0, 0, 0, 1]])
  reset_matrix = np.array([[1, 0, 0, -o_x], [0, 1, 0, -o_y],
                           [0, 0, 1, -o_z],[0, 0, 0, 1]])
  transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
  return transform_matrix


def pad_volume (arr, padding = 5):
    """padding in pixels"""
    sz = arr.shape
    #padding along 0th axis
    p_arr = np.zeros(shape=(padding,sz[1],sz[2]), dtype=arr.dtype)
    ans = np.concatenate([p_arr,arr,p_arr],axis=0)
    #padding along 1st axis
    p_arr = np.zeros(shape=(sz[0]+2*padding,padding,sz[2]),dtype=arr.dtype)
    ans = np.concatenate([p_arr,ans,p_arr],axis=1)
    #padding along 2nd axis
    p_arr = np.zeros(shape=(sz[0]+2*padding,sz[1]+2*padding,padding),
                            dtype=arr.dtype)
    ans = np.concatenate([p_arr,ans,p_arr],axis=2)

    return ans

def reduce_dim (arr, channel=0):
    """channel is the channel to be preserved"""
    return arr[...,channel]



def translate_volume (vol,dr):
    """"translates a volume by dr=[dx,dy,dz].
    """
    print ("In translate_volume")
    print ("Shape",vol.shape)
#    dr.append(1)# [dx,dy,dz,1]
#    mat = np.zeros(shape=(4,4),dtype = np.float32)
#    mat[0,0]=1
#    mat[1,1]=1
#    mat[2,2]=1
#    mat[3,3] = 1
#    mat[0,1] = 0
#    mat[0,2] = 0
#    for i in range(3):
#        mat[i,3] = dr[i]
#    print ("T",mat)
    x,y,z= vol.shape
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    o_z = float(z) / 2 + 0.5
#    print("Coords",x,y,z,_)
#    mat = transform_matrix_offset_center(mat,x,y,z)
    theta = 0.1#rad
    c = np.cos(theta)
    s = np.sin(theta)
    mat = np.array([[1, 0, 0 ],[0, c , s ],[0, -s, c]],dtype=np.float32)
#    ans = affine_transform(vol,mat,order=0,offset=[o_x,o_y,o_z])
    ans = shift(vol,[dr[0],dr[1],dr[2]])
    np.clip(ans,0.001,0.999,ans)
    ut.get_array_info(ans,"ans")
    return ans

def run_test ():
    mat = np.linspace(1,20,20,dtype=np.float32).reshape((4,5))
    mat2 = np.zeros(shape=(4,5),dtype=np.float32)
    print (mat)
    print (mat2)
    x = np.array([0,1],dtype=np.int32)
    y = x+2
    print ("x:",x)
    print ("y:",y)
    mat2[y,y] =  mat[x,x]
    print (mat2)

if __name__ == "__main__":
#%%
#    Y = translate_volume(X[0,:,:,:,0],[0,0,0])
#    print(Y.shape)
#    Y = np.expand_dims(Y,0)
#    Y = np.expand_dims(Y,-1)
#    X = np.concatenate([X,Y,X],axis=0)
#    vv= VV(X)
#%%
#    X= get_mesh(16,[16,16,16])
    X = X.reshape(X.shape[1], X.shape[2], X.shape[3])
    Y = translate_volume(X,[0,20,-20])
    print(Y.shape)
    Y = np.expand_dims(Y,0)
    Y = np.expand_dims(Y,-1)
    X = np.expand_dims(X,0)
    X = np.expand_dims(X,-1)
    X = np.concatenate([X,Y,X],axis=0)
    vv= VV(X)


#%%