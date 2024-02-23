#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 16:33:48 2017

@author: shubham
"""

import numpy as np
import matplotlib.pyplot as plt
import data as dt
import utility as ut
import matplotlib.pyplot as plt
from skimage import data
import matplotlib.cm as cm

def get_n_channel_image (arr,n=3):
    """"Returns an n-channeled image. Copies and concatenates the given
        array along the last axis.
        shape of the input arr must be of the form (s1,s2,...,1)
    """
    if n==0:
        return np.squeeze(arr, axis=-1)
    c_list = []
    assert(arr.shape[-1]==1)
    for i in range(n):
        c_list.append(arr)
    return np.concatenate(c_list,axis=-1)
class VolumeViewer:
    def __init__ (self,dataset, cmap):
#        if  (dataset.shape[-1]==1):
#            print("Converting to 3 Channel image.")
#            dataset = get_n_channel_image(dataset,3)
        self.cmap = cmap
        self.n_channels = 0
        self.mri_dataset_viewer(dataset)
        


    def mri_dataset_viewer (self,dataset):
        self.fig, self.ax = plt.subplots()
        self.ax.dset = dataset
        self.ax.dsz = dataset.shape[0]
        self.ax.sample = 0
        print("current_sample:",self.ax.sample)
        volume = self.ax.dset[self.ax.sample]
        self.ax.volume = volume
        self.ax.index = volume.shape[0] // 2
        self.ax.imshow(get_n_channel_image(volume[self.ax.index],self.n_channels), cmap=self.cmap)
        self.ax.axis = 0
        self.fig.canvas.mpl_connect('key_press_event', self.process_key)
        plt.show()

    def process_key(self,event):
        self.fig = event.canvas.figure
        self.ax = self.fig.axes[0]
        if event.key == 'h':
            self.previous_slice(self.ax)
        elif event.key == 'j':
            self.next_slice(self.ax)
        elif event.key == 'u':
            self.next_sample(self.ax)
        elif event.key == 'y':
            self.previous_sample(self.ax)
        elif event.key == 't':
            self.change_axis (self.ax)
        self.fig.canvas.draw()

    def previous_slice(self,ax):
        """Go to the previous slice."""
        volume = ax.volume
        ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
        ax.images[0].set_array(get_n_channel_image(volume[ax.index],self.n_channels))
        print("previous slice")

    def next_slice(self,ax):
        """Go to the next slice."""
        volume = ax.volume
        ax.index = (ax.index + 1) % volume.shape[0]
        ax.images[0].set_array(get_n_channel_image(volume[ax.index],self.n_channels))
        print("slice", ax.index)

    def next_sample (self,ax):
        ax.axis = 0
        ut.get_array_info(ax.volume,"Vol_o")
        ax.sample = (ax.sample+1)%ax.dsz
        print("sample num:",ax.sample)
        ax.volume = ax.dset[ax.sample]
        ax.images[0].set_array(get_n_channel_image(ax.volume[ax.index],self.n_channels))
        ut.get_array_info(ax.volume,"Vol_1")

    def previous_sample (self,ax):
        print("prev sample")
        ax.axis = 0
        ax.sample = (ax.sample-1)%ax.dsz
        ut.get_array_info(ax.volume,"Vol_o")
        ax.volume = ax.dset[ax.sample]
        ax.images[0].set_array(get_n_channel_image(ax.volume[ax.index],self.n_channels))
        ut.get_array_info(ax.volume,"Vol_1")

    def change_axis (self,ax):
#        print("prev sample")
        ax.axis = (ax.axis+1)%3
#        ut.get_array_info(ax.volume,"Vol_o")
        ax.volume = np.transpose(ax.volume,(2,0,1,3))
        ax.images[0].set_array(get_n_channel_image(ax.volume[ax.index],self.n_channels))
        print ("Current axis:",ax.axis)
#        ut.get_array_info(ax.volume,"Vol_1")

if __name__ == '__main__':
    import data as dt
    X, Y = dt.get_newext_dev_parkinson_cls_data()
    Z, = dt.get_newext_test_parkinson_cls_data()
    ut.get_array_info(X, "Training")
    ut.get_array_info(Z, "Test")
    cmap_to_use = "YlOrRd"
    # vv =VolumeViewer(X, cm.get_cmap(cmap_to_use))
    vv =VolumeViewer(Z, cm.get_cmap(cmap_to_use))
    """
    Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r,
     BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r,
    Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, 
    Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, 
    Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r,
    PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r,
    RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r,
    RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r,
    Set3, Set3_r, Spectral, Spectral_r, Vega10, Vega10_r, Vega20,
    Vega20_r, Vega20b, Vega20b_r, Vega20c, Vega20c_r, Wistia, Wistia_r,
    YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, 
    afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, 
    brg, brg_r, bwr, bwr_r, cool, cool_r, coolwarm, coolwarm_r, copper,
    copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r,
    gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r,
    gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, 
    gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r,
    hsv, hsv_r, inferno, inferno_r, jet, jet_r, magma, magma_r, nipy_spectral,
    nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, 
    prism, prism_r, rainbow, rainbow_r, seismic, seismic_r, spectral, spectral_r,
    spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b,
    tab20b_r, tab20c, tab20c_r, terrain, terrain_r, viridis, viridis_r, winter, winter_r
    """


