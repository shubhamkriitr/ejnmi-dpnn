#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 06:02:01 2017

@author: shubham
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import shutil
import warnings

# FOLDER+FILE OPS
def does_exist (dir_path,file_or_folder_name):
    dir_data = os.listdir(dir_path)
    # print (dir_data)
    if file_or_folder_name in dir_data:
        print(file_or_folder_name+"is present in "+dir_path)
        return True
    print(file_or_folder_name+"is not in "+dir_path)
    return False

def create_if_does_not_exist (dir_path,file_or_folder_name,content=None,
                              assume_folder_if_without_dot = True):
    dir_data = os.listdir(dir_path)
    if file_or_folder_name in dir_data:
        return False
    else:
        if ("." not in file_or_folder_name and assume_folder_if_without_dot and
            content==None):
            os.mkdir(dir_path+os.sep+file_or_folder_name)
        else:
            with open(dir_path+os.sep+file_or_folder_name,'w') as f:
                if (content is not None):
                    assert(isinstance(content,str))
                    f.write(content)
                else:
                    pass
    return True

def add_file_or_folder (dir_path, file_or_folder_name,
                        append_number_if_entity_already_exists=False):
    if (does_exist(dir_path,file_or_folder_name)) and not append_number_if_entity_already_exists:
        raise ValueError("File already exists. Set the flag to True to allow\
                             appending numbers to the file/folder name.")
    else:
        name = file_or_folder_name
        i = 0
        while (not create_if_does_not_exist(dir_path,name)):
            i+=1
            name = file_or_folder_name+"_"+str(i)
        print("Created:",dir_path+os.sep+name)
        return dir_path+os.sep+name

def get_ancestor (cwd, level):
    """returns ancestor folder at distance level from the
       current working directory.e.g. level=1 --> parent
    """
    anc_list = cwd.split(os.sep)
    anc = ""
    for i in range(1,len(anc_list)-level):
        anc = anc+os.sep+anc_list[i]
    return anc

def get_array_info (arr,name="Array"):
    print("-"*10)
    print (name,"specifications:")
    print (type(arr))
    print (arr.shape)
    print (arr.dtype)
    print ("Mean",np.mean(arr))
    print ("Median",np.median(arr))
    print ("Max",np.max(arr))
    print ("Min",np.min(arr))
    print ("std.dev.",np.std(arr))
    print("-"*10)

def get_n_channel_image (arr,n=3):
    """"Returns an n-channeled image. Copies and concatenates the given
        array along the last axis.
        shape of the input arr must be of the form (s1,s2,...,1)
    """
    c_list = []
    assert(arr.shape[-1]==1)
    for i in range(n):
        c_list.append(arr)
    return np.concatenate(c_list,axis=-1)

def append_time_string (input_str):
    return input_str + time.strftime("%Y-%m-%d-%H%M%S")


def create_log_file(file_path_or_obj, list_of_items=[], level=0):
    """Creates a text file, with path file_path,
       containing all the items in the list_of_items
    """
    sep = "="*20 + "\n"
    if level == 0:
        assert (type(list_of_items) is list)
        f =  open(file_path_or_obj, "w")
    else:
        f = file_path_or_obj

    pfix = level*"--->"
    for item in list_of_items:
        if level==0:
            f.write(sep)
        if isinstance(item, dict):
            f.write(pfix+"---------\n")
            for key in item.keys():
                f.write("    "+pfix+str(key)+":\n")
                if isinstance(item[key], np.ndarray):
                    f.write("    "+pfix+"numpy.ndarray --"+str(item[key].shape)
                            + "--" + str(item[key].dtype) + "\n")
                elif isinstance(item[key], list):
                    create_log_file(f,item[key],level+1)
                else:
                    f.write("    "+pfix+str(item[key])+"\n")
        else:
            if isinstance(item, np.ndarray):
                f.write(pfix+"numpy.ndarray --"+str(item.shape)
                        + "--" + str(item.dtype) + "\n")
            elif isinstance(item, list):
                    create_log_file(f,item,level+1)
            else:
                f.write(pfix+str(item)+"\n")
    if level==0:
        f.close()

def view_volume (dataset, sample_number,ch_dim):
    """dataset must be of the shape [N, H, W, D]"""
    fig = plt.figure()
    X = dataset
    def animate(i):
        if ch_dim == 1:
            image = X[sample_number,i,:,:]
        elif ch_dim == 2:
            image = X[sample_number,:,i,:]
        elif ch_dim == 3:
            image = X[sample_number,:,:,i]
        plt.imshow(image)
    ani = animation.FuncAnimation(fig, animate, interval=69)
    plt.show()

def get_list_difference (L1,L2,L=None):
    """Returns L1 - (L1 intersection L2)
    Args:
        L1 (list): List 1
        L2 (list): List 2
        L (optional,list): result is appended to this list and returned
    Returns:
        A list containing all the elements in L1 which are not in L2.
    """
    ans = []
    if L is not None:
        ans = L
    for item in L1:
        if item not in L2:
            ans.append(item)
    return ans

def print_sequence (seq):
    i = 1
    for item in seq:
        print("Sr. {}: ".format(i), item)
        i+=1


def get_chunk_ranges (range_ ,fold_number,max_fold=5):
    a = range_[0]
    b = range_[1]
    sz = b - a + 1
    rem = sz%max_fold
    cs = int (sz/max_fold)
    train = []
    val = []
    j = a
    for i in range(1,max_fold+1):
        csz = cs #chunk size
        if rem != 0:
            rem-=1
            csz+=1
        r= (j, j + csz -1)  # as r[1] is inclusive
        if i==fold_number:
            val.append(r)
        else:
            train.append(r)
        j = j + csz
    return {"train":train,"val":val}

def get_split_ranges (ranges,fold_number,max_fold=5):
    train = []
    val = []
    for item in ranges:
        split = get_chunk_ranges(item,fold_number,max_fold)
        for r in split["train"]:
            train.append(r)
        for r in split["val"]:
            val.append(r)
    return {"train":train,"val":val}

def find_paths (dir_path, key_words,level=1):
    """returns a list of paths(strings) to files or folders
    which are at a depth=`level` from the `dir_path`.
    NOTE: Here it is assumed that names containing
    """
    paths = []
    if level==1:
        for item in os.listdir(dir_path):
            flag = True
            for key_word in key_words:
                if key_word not in item:
                    flag = False
                    break
            if flag:
                paths.append(dir_path+os.sep+item)
    elif level>1:
        for item in os.listdir(dir_path):
            try:
                for p in find_paths(dir_path+os.sep+item,key_words,level-1):
                    paths.append(p)
            except NotADirectoryError:
                pass
    else:
        raise ValueError(
        "level must be >= 1. Given value of level is {}".format(level))
    return paths


def find_and_copy (src_dir, op_dir,set_id,match_terms,level,append_time_stamp):
    files = find_paths(src_dir,match_terms,level)
    #print("Found",files)#debug_

    if len(files)==0:
        warnings.warn("No files with match terms:"+str(match_terms)+"are found. ")
        return

    if not os.path.exists(op_dir):
        os.mkdir(op_dir)

    if set_id is not None:
        op_path = op_dir+os.sep+set_id
        if append_time_stamp:
            op_path = append_time_string(op_path+"_")
        if not os.path.exists(op_path):
            os.mkdir(op_path)
    else:
        op_path = op_dir


    for src_file in files:
        dest = src_file.split(os.sep)
        dest = op_path + os.sep + dest[-1]
        shutil.copy(src_file,dest)
        print("copied {} --> {}".format(src_file,dest))


class Timer ():
    def __init__(self):
        self.created_at = time.time()
        self.base_time = None
        self.started = False
    def start(self):
        if not self.started:
            self.base_time = time.time()
        else:
            print("Timer already running..")
    def elapsed(self):
        el = (int) (time.time()- self.base_time)
        return "{}:{}:{}".format(int(el/3600),
                int(el/60),el%60
                )


if __name__ == '__main__':
    CWD = os.getcwd()
    PD = os.path.abspath(os.pardir)
    print(PD)
    if not does_exist(CWD,"Summaries"):
        print("Ok Creating...")
        create_if_does_not_exist(CWD,"Summaries")
    if does_exist(CWD,"Summaries"):
        print("Created.")

    add_file_or_folder(CWD,"T_dir",True)
    add_file_or_folder(CWD,"T_dir",True)
    add_file_or_folder(CWD,"T_dir",True)
    add_file_or_folder(CWD,"T_dir",True)
    add_file_or_folder(CWD,"T_dir",True)



