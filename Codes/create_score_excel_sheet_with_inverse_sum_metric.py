import numpy as np
import utility as ut
import os
import performance_calculator_bin_search as pc
from copy import deepcopy
import fold_wise_bar_plot as fwbp
import time
import pandas as pd
import pdb
import h5py as hf

"""
1. Store models-evaluation-h5 file in the following dir structure
ROOT
|
----Selected_<tagsep>SET_<set number>_epoch_<epoch_numebr><tagsep>_time
    |
    ---SOME_FODLER
        |
        ---scores
            |
            ---<model_name>_fold_<number>score.h5()# uniquely idendified\
            by <model_name> fold_<fold_number> and score.h5(Pass it as list `file_tag_list`)
                |->val/[MSA_xls|PD_xls|PSP_xls]
"""


def create_score_sheet(root_folder, tagsep, file_tag_list=["score.h5"], fold_list=[1,2,3,4,5], dtype=np.float32, xl_file_prefix="compiled_result"):
    columns = ["SET", "EPOCH", "FOLD", ]
    cl = {0:"MSA", 1:"PSP", 2:"PD"} # DO NOT CHANGE
    mtrc = {0:"TPR", 1:"TNR", 2:"PPV", 3:"NPV", 4:"INV_SUM",5:"PRODUCT",6:"F1_SCORE"}
    for cl_idx in range(3):
        for mt_idx in range(7):
            columns.append(cl[cl_idx]+"_"+mtrc[mt_idx])
    files = get_h5_score_file_list(root_folder, tagsep, file_tag_list, fold_list)
    print(files)
    s = list(sorted(files.keys()))
    score_arr = np.zeros(shape=(len(s), 24))
    print(s)
    i=-1
    for tag in s:
        i+=1
        print("For:"+tag)
        ids = tag.split("_")
        id_col_data = [int(ids[1]),int(ids[3]),int(ids[5])]
        score_row = get_score_array(id_col_data, files[tag], dtype)
        score_arr[i] = score_row
        print(score_row)
    print(score_arr)
    df = pd.DataFrame(score_arr, columns=columns)
    df.to_excel(ut.append_time_string(xl_file_prefix)+".xls")
    # pdb.set_trace()


def get_score_array(id_column_data, score_hdf_file_loc, dtype=np.float32):
    """
    each item in id_column_data will be store in separate column,
    size of output array will be (1, len(id_column_data)+3*4)
    """
    nm = 7# num of metrics
    x = np.zeros(shape=(1, len(id_column_data)+3*nm))
    for i in range(len(id_column_data)):
        x[0,i] = id_column_data[i]
    with hf.File(score_hdf_file_loc, 'r') as f:
        scrs = []
        cl = {0:"MSA", 1:"PSP", 2:"PD"} # DO NOT CHANGE
        for cl_idx in range(3):
            # pdb.set_trace()
            scr_d = f["val/"+cl[cl_idx]+"_xls"][:]
            scr_d = (scr_d).astype(dtype)
            scr_data = np.zeros(shape=(1, nm), dtype=dtype)
            inv_sum = np.sum(1/(scr_d+1e-7))
            prod = np.prod(scr_d)
            tpr=scr_d[0,0]
            ppv = scr_d[0,2]
            f1 = (2*(tpr*ppv))/(tpr+ppv+1e-7)
            scr_data[:, 0:4] = scr_d
            scr_data[:, 4] = inv_sum
            scr_data[:,  5] = prod
            scr_data[:, 6] = f1
            scrs.append(scr_data)
        score_row = np.concatenate(scrs, axis=1)
    x[:, len(id_column_data):]=score_row
    # pdb.set_trace()
    return x


def get_h5_score_file_list(root_folder, tagsep, file_tag_list=["score.h5"], fold_list=[1,2,3,4,5]):
    """
    Returns a dictionary of absolute file paths idenified by tag,
    where tag = <set_number>_epoch_<number>_fold_<fold_number>
    """
    L = os.listdir(root_folder)
    L.sort()
    file_dict = {}
    for d in L:
        tag_1 = ""
        p = root_folder + os.sep + d
        tag_1 = p.split(tagsep)[1] # SET_1_epoch_200
        f_list = ut.find_paths(p,file_tag_list, 3)
        for fold in fold_list:
            for fname in f_list:
                if ("fold_"+str(fold)) in fname:
                    tag = tag_1+"_fold_"+str(fold)
                    assert(tag not in file_dict.keys())
                    file_dict[tag] = fname
    return file_dict


        
if __name__ == '__main__':
    rf = "/home/abhijit/nas_drive/Abhijit/Shubham/ejnmmi-dpnn/Codes/Checkpoints/ALL_BULK_SET_19"
    tagsep = "_tgx_"
    create_score_sheet(rf,tagsep)
        
