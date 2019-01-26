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

class VolumeViewer:
    def __init__ (self,dataset):
#        if  (dataset.shape[-1]==1):
#            print("Converting to 3 Channel image.")
#            dataset = ut.get_n_channel_image(dataset,3)
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
        self.ax.imshow(ut.get_n_channel_image(volume[self.ax.index],3))
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
        ax.images[0].set_array(ut.get_n_channel_image(volume[ax.index],3))
        print("previous slice")

    def next_slice(self,ax):
        """Go to the next slice."""
        volume = ax.volume
        ax.index = (ax.index + 1) % volume.shape[0]
        ax.images[0].set_array(ut.get_n_channel_image(volume[ax.index],3))
        print("slice", ax.index)

    def next_sample (self,ax):
        ax.axis = 0
        ut.get_array_info(ax.volume,"Vol_o")
        ax.sample = (ax.sample+1)%ax.dsz
        print("sample num:",ax.sample)
        ax.volume = ax.dset[ax.sample]
        ax.images[0].set_array(ut.get_n_channel_image(ax.volume[ax.index],3))
        ut.get_array_info(ax.volume,"Vol_1")

    def previous_sample (self,ax):
        print("prev sample")
        ax.axis = 0
        ax.sample = (ax.sample-1)%ax.dsz
        ut.get_array_info(ax.volume,"Vol_o")
        ax.volume = ax.dset[ax.sample]
        ax.images[0].set_array(ut.get_n_channel_image(ax.volume[ax.index],3))
        ut.get_array_info(ax.volume,"Vol_1")

    def change_axis (self,ax):
#        print("prev sample")
        ax.axis = (ax.axis+1)%3
#        ut.get_array_info(ax.volume,"Vol_o")
        ax.volume = np.transpose(ax.volume,(2,0,1,3))
        ax.images[0].set_array(ut.get_n_channel_image(ax.volume[ax.index],3))
        print ("Current axis:",ax.axis)
#        ut.get_array_info(ax.volume,"Vol_1")

if __name__ == '__main__':
    X = VolumeViewer(arr)
