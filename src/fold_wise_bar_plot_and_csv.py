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
        ax.text(rect.get_x() + rect.get_width()/2., 1.02*height,
                '%10.2f' % height,
                ha='center', va='bottom',color="blue",fontsize=14)#bbox=dict(facecolor='white', alpha=0.5))


def _bar_plot (ax,array,color='r',shift=0.0, width=0.35,title="BarPlot",
               xlabel=None,ylabel="%",xticks=None,yticks=None):
    ind = np.arange(array.shape[0])
    print("ind",ind)
    rects = ax.bar(ind + shift, array, width, align='edge',color=color)
    _auto_label(ax,rects)
    ax.set_title(title,fontsize=14)
    ax.set_ylim([0,110])
    ax.set_yticklabels([0,20,40,60,80,100],fontsize=14)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_xticks(ind)
    if xticks is not None:
        n = array.shape[0] - len(xticks)
        for i in range(n):
            xticks.append("N/A")
        ax.set_xticklabels(xticks,fontsize=14)
    if yticks is not None:
        n = array.shape[0] - len(yticks)
        for i in range(n):
            yticks.append("N/A")
        ax.set_yticklabels(yticks,fontsize=14)




def generate_plots (input_folder,op_folder,num_files=5,
                    key_term="fold", suffix="plots"):
    names = os.listdir(input_folder)
    cls = {0:'MSA', 1:'PSP', 2:'PD'}
    xticks = ["TPR", "TNR", "PPV", "NPV"]
    avg_val_cfmat = np.zeros(shape=(3,2,2),dtype=np.float32)
    avg_train_cfmat = np.zeros(shape=(3,2,2),dtype=np.float32)
    avg_xl_arr = np.zeros((1,12),dtype=np.float32)
    N=0
    for i in range (1,num_files+1):
        term = key_term+"_"+str(i)
        for fn in names:
            if term in fn and ".h5" in fn:
                f_loc = input_folder+os.sep+fn
                with hf.File(f_loc,"r") as f:
                    fig, axs = plt.subplots(2,3,figsize=(18,12))
                    fig.subplots_adjust(top=0.2)
                    xl_arr = np.zeros((1,12),dtype=np.float32)
                    for k in range(3):
                        arr = f["train/"+cls[k]+"_xls"][0]
                        arr = arr*100
                        title = cls[k]+"(Training[{}])".format(i)
                        _bar_plot(axs[0][k],arr,title=title,
                                  xticks=xticks)
                        avg_train_cfmat[k] = avg_train_cfmat[k] + f["train/"+cls[k]+"_cfmat"][:]
                    for k in range(3):
                        arr = f["val/"+cls[k]+"_xls"][0]
                        arr = arr*100
                        xl_arr[0,4*k:(4*k+4)] = arr
                        title = cls[k]+"(Validation[{}])".format(i)
                        _bar_plot(axs[1][k],arr,title=title,
                                  xticks=xticks)
                        avg_val_cfmat[k] = avg_val_cfmat[k] + f["val/"+cls[k]+"_cfmat"][:]
                    plt.tight_layout(1.2,1.2,1.4)
#                    plt.autoscale()
                    plt.savefig(op_folder+os.sep+suffix+"_"+str(i)+".png")
                    np.savetxt(op_folder+os.sep+suffix+"_"+str(i)+".csv",xl_arr,delimiter=",")
                    avg_xl_arr = avg_xl_arr + xl_arr
                    plt.close('all')
                    N+=1
    fig, axs = plt.subplots(2,3,figsize=(18,12))
    avg_xl_arr = avg_xl_arr/N
    np.savetxt(op_folder+os.sep+suffix+"_avg_"+".csv",avg_xl_arr,delimiter=",")
    for k in range(3):
        title = cls[k]+"(Validation)"
        print(k,avg_val_cfmat[k])
        print(k,avg_train_cfmat[k])
        arr = get_scores_from_cfmat(avg_val_cfmat[k])
        arr = arr*100
        _bar_plot(axs[1][k],arr,title=title,
                  xticks=xticks)
        title = cls[k]+"(Training)"
        arr = get_scores_from_cfmat(avg_train_cfmat[k])
        arr = arr*100
        _bar_plot(axs[0][k],arr,title=title,
                  xticks=xticks)
        plt.tight_layout(1.2,1.2,1.4)
        plt.savefig(op_folder+os.sep+"average_score_"+suffix+"_"+".png")
    
    fig, axs = plt.subplots(1,3,figsize=(18,12))
    for k in range(3):
        title = cls[k]+"(Validation)"
        arr = avg_xl_arr[0,4*k:(4*k+4)]
        _bar_plot(axs[k],arr,title=title,
                  xticks=xticks)
        plt.tight_layout(1.2,1.2,1.4)
        plt.savefig(op_folder+os.sep+"CORRECT_AVG_"+suffix+"_"+".png")



if __name__ == "__main__":
#%%
#    ifn = "/home/shubham/Desktop/ENIGMA_CODES/Master/Final_scores_and_plots/3d-net-eval-Outputs-2"
#    generate_plots(ifn,ifn,5)
#    plt.close('all')
#    ifn = "/home/shubham/Desktop/ENIGMA_CODES/Master/Final_scores_and_plots/Proj-FCN-Eval-Outputs-2"
#    generate_plots(ifn,ifn,5)
#    plt.close('all')
#    ifn = "/home/shubham/Desktop/ENIGMA_CODES/Master/Final_scores_and_plots/SET-2-Avg-Pooled-Eval-Outputs-2"
#    generate_plots(ifn,ifn,5)
#    plt.close('all')

#%%
    ifn = "/home/shubham/Desktop/ENIGMA_CODES/Master/Final_scores_and_plots/1_3D-ConvNet"
    generate_plots(ifn,ifn,5)
    plt.close('all')
    ifn = "/home/shubham/Desktop/ENIGMA_CODES/Master/Final_scores_and_plots/2_Projection_Net_Avg_Pool"
    generate_plots(ifn,ifn,5)
    plt.close('all')
    ifn = "/home/shubham/Desktop/ENIGMA_CODES/Master/Final_scores_and_plots/3_Projection_Net_FCN"
    generate_plots(ifn,ifn,5)
    plt.close('all')
    wait  = input("wait")
#%%
    L = [1,7,8,9]
    ifn_1 = "/home/shubham/Desktop/ENIGMA_CODES/Master/Final_scores_and_plots/Best  Ones-Eval/EXP_13_SET_1-Outputs-2"
    for i in L:
        ifn = os.getcwd()+os.sep+"Best  Ones-Eval"+os.sep+"EXP_13_SET_"+str(i)+"-Outputs-2"
#        print(ifn)
#        print(ifn_1)
#        assert(ifn==ifn_1)
#        op_dir = os.getcwd()+os.sep+"Best Ones-Eval"+os.sep+"EXP_13_SET_"+str(i)+"-Plots"
#        try:
#            os.mkdir(op_dir)
#        except FileExistsError:
#            op_dir = ifn
        generate_plots(ifn,ifn,5)



    x = np.array([[143, 2],[2, 263]],dtype=np.float32)
    print(get_scores_from_cfmat(x))

