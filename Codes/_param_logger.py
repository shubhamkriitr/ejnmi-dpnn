#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 03:37:38 2018

@author: shubham
"""
from skimage import data
from skimage.viewer import ImageViewer
import matplotlib.pyplot as plt
import data

l = [(0,90), (91,120), (121,256)]
r = []
s = []
split = 0.2
i=-1
for x in l:
    i+=1
    r.append([x[0], int(x[0]+(x[1]-x[0])*split)])
    s.append([r[i][1]+1,x[1]])
print(r,s)
X , Y = data.get_parkinson_data(ranges=r)
X=X[:,:,:,:,0]
#import _visualize
#V = _visualize.VolumeViewer(X)
#X = np.transpose(X,[0,1,3,2])
Y = Y[:,:,0]
#X_val, Y_val = data.get_parkinson_data(ranges=s)
#X_val = X_val[:,:,:,:,0]


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

def view_volume (dataset, sample_number,ch_dim):
    """dataset must be of the shape [N, H, W, D]"""
    fig = plt.figure()
    X = dataset
    def animate(i):
        if i<X.shape[ch_dim]:
            if ch_dim == 1:
                image = X[sample_number,i,:,:]
            elif ch_dim == 2:
                image = X[sample_number,:,i,:]
            elif ch_dim == 3:
                image = X[sample_number,:,:,i]
            plt.imshow(image)
    ani = animation.FuncAnimation(fig, animate, interval=10)
    plt.show()




import utility
utility.view_volume(X,0,1)











