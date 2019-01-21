#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 23:37:10 2018

@author: shubham
"""

import matplotlib.pyplot as plt
import numpy as np

"""
========
Barchart
========

A bar plot with errorbars and height labels on individual bars
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py as hf

def get_scores_from_cfmat (cfmat):
    TP = cfmat[0,0]
    FP = cfmat[0,1]
    FN = cfmat[1,0]
    TN = cfmat[1,1]
    TPR, TNR, PPV, NPV = 0,0,0,0
    if (TP + FN) > 0:
        TPR = TP/(TP + FN)
    if (FP + TN) > 0:
        TNR = TN/(FP + TN)
    if (TP + FP) > 0:
        PPV = TP/(TP + FP)
    if (TN + FN) > 0:
        NPV = TN/(TN + FN)
    return np.array([TPR, TNR, PPV, NPV],dtype=np.float32)

def _auto_label(ax, rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.0*height,
                '%10.2f' % height,
                ha='center', va='bottom')


def _bar_plot (ax,array,color='r',shift=0.0, width=0.35,title="BarPlot",
               xlabel=None,ylabel="%",xticks=None,yticks=None):
    ind = np.arange(array.shape[0])
    print("ind",ind)
    rects = ax.bar(ind + shift, array, width, color=color)
    _auto_label(ax,rects)
    ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_xticks(ind)
    if xticks is not None:
        n = array.shape[0] - len(xticks)
        for i in range(n):
            xticks.append("N/A")
        ax.set_xticklabels(xticks)
    if yticks is not None:
        n = array.shape[0] - len(yticks)
        for i in range(n):
            yticks.append("N/A")
        ax.set_yticklabels(yticks)




def generate_plots (input_folder,op_folder,num_files=5,
                    key_term="fold", suffix="plots"):
    names = os.listdir(input_folder)
    cls = {0:'MSA', 1:'PSP', 2:'PD'}
    xticks = ["TPR", "TNR", "PPV", "NPV"]
    avg_val_cfmat = np.zeros(shape=(3,2,2),dtype=np.float32)
    avg_train_cfmat = np.zeros(shape=(3,2,2),dtype=np.float32)

    for i in range (1,num_files+1):
        term = key_term+"_"+str(i)
        for fn in names:
            if term in fn and ".h5" in fn:
                f_loc = input_folder+os.sep+fn
                with hf.File(f_loc,"r") as f:
                    fig, axs = plt.subplots(3,2,figsize=(12,12))
                    for k in range(3):
                        arr = f["train/"+cls[k]+"_xls"][0]
                        arr = arr*100
                        title = cls[k]+"(Training[{}])".format(i)
                        _bar_plot(axs[k][0],arr,title=title,
                                  xticks=xticks)
                        avg_train_cfmat[k] = avg_train_cfmat[k] + f["train/"+cls[k]+"_cfmat"][:]
                    for k in range(3):
                        arr = f["val/"+cls[k]+"_xls"][0]
                        arr = arr*100
                        title = cls[k]+"(Validation[{}])".format(i)
                        _bar_plot(axs[k][1],arr,title=title,
                                  xticks=xticks)
                        avg_val_cfmat[k] = avg_val_cfmat[k] + f["val/"+cls[k]+"_cfmat"][:]
                    plt.tight_layout(1.2,1.2,1.4)
                    plt.savefig(op_folder+os.sep+suffix+"_"+str(i)+".png")
                    plt.close('all')
    fig, axs = plt.subplots(3,2,figsize=(10,10))
    for k in range(3):
        title = cls[k]+"(Validation)"
        print(k,avg_val_cfmat[k])
        print(k,avg_train_cfmat[k])
        arr = get_scores_from_cfmat(avg_val_cfmat[k])
        arr = arr*100
        _bar_plot(axs[k][1],arr,title=title,
                  xticks=xticks)
        title = cls[k]+"(Training)"
        arr = get_scores_from_cfmat(avg_train_cfmat[k])
        arr = arr*100
        _bar_plot(axs[k][0],arr,title=title,
                  xticks=xticks)
        plt.tight_layout(1.2,1.2,1.4)
        plt.savefig(op_folder+os.sep+"average_score_"+suffix+"_"+".png")



if __name__ == "__main__":
#    ifn = "/home/shubham/Desktop/ENIGMA_CODES/Master/Final_scores_and_plots/3d-net-eval-Outputs-2"
#    generate_plots(ifn,ifn,5)
#    plt.close('all')
    ifn = "/home/shubham/Desktop/ENIGMA_CODES/Master/Final_scores_and_plots/Proj-FCN-Eval-Outputs-2"
    generate_plots(ifn,ifn,5)
    plt.close('all')
#    ifn = "/home/shubham/Desktop/ENIGMA_CODES/Master/Final_scores_and_plots/SET-2-Avg-Pooled-Eval-Outputs-2"
#    generate_plots(ifn,ifn,5)
#    plt.close('all')
    x = np.array([[143, 2],[2, 263]],dtype=np.float32)
    print(get_scores_from_cfmat(x))

