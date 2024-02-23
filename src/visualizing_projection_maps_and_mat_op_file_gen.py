import pandas
import numpy as np
import matplotlib.pyplot as plt
import data as dt
import utility as ut
import matplotlib.cm as cm
import h5py as hf
import scipy.io as sio
import os





def save_projections_as_mat(input_h5_file, idx_to_name_map, output_mat_location):
    with hf.File(input_h5_file, 'r') as f:
        arr = f['test']['projections']
        print("Input Arr: ", arr.shape)
        arr_dict = {}
        for ix in range(arr.shape[0]):
            arr_dict[idx_to_name_map[ix]] = arr[ix,:,:,0]
        sio.savemat(output_mat_location, arr_dict)


def plot_and_save(arr, cmap_name, op_loc):
    plt.close('all')
    fig, ax = plt.subplots(1,1, figsize=(8, 8 ))
    im = ax.imshow(arr, cmap=cm.get_cmap(cmap_name))
    fig.colorbar(im, ax=ax)
    plt.savefig(op_loc)
    # plt.show()

def generate_and_save_maps_single_cmap(image_arr_dict, image_name_list, output_dir, cmap_name):
    for img_name in image_name_list:
        op_loc = output_dir+os.sep+img_name+".png"
        plot_and_save(image_arr_dict[img_name], cmap_name, op_loc)

def generate_and_save_maps(image_arr_dict, image_name_list, output_dir, cmap_name_list):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for cmap_name in cmap_name_list:
        op_sub_dir_name = "color_scheme__"+cmap_name
        dir_DNE = ut.create_if_does_not_exist(output_dir, op_sub_dir_name)
        if not dir_DNE:
            print('Overwriting: Directory "{}" in "{}" already exists.'.format(cmap_name, output_dir ))
        op_sub_dir = output_dir+os.sep+op_sub_dir_name
        generate_and_save_maps_single_cmap(image_arr_dict, image_name_list, op_sub_dir, cmap_name)


if __name__ == '__main__':
    root_dir = '/home/abhijit/nas_drive/Abhijit/Shubham/ejnmmi-dpnn/Codes/ProjectionVisualization'
    floc = '/home/abhijit/nas_drive/Abhijit/Shubham/ejnmmi-dpnn/Codes/ProjectionVisualization/SUB3_NEW_EXT_DATASET/PXNET_GAP_SIG_WTD_NP_TEST_RESULT_on_training_dataSET_15_18_19_20_21_train_dataensemble__predictions.h5'
    mat_oploc = '/home/abhijit/nas_drive/Abhijit/Shubham/ejnmmi-dpnn/Codes/ProjectionVisualization/SUB3_NEW_EXT_DATASET/new_ext_training_data_Projections.mat'
    figure_opdir = '/home/abhijit/nas_drive/Abhijit/Shubham/ejnmmi-dpnn/Codes/ProjectionVisualization/SUB3_NEW_EXT_DATASET/TRAINING_DATA_FIGURES'
    training_idx_to_name = '/home/abhijit/nas_drive/Abhijit/Shubham/ejnmmi-dpnn/Codes/new_ext_train_data_idx_to_vol_mapping.txt'
    
    with open(training_idx_to_name, 'r') as f:
        idx_to_name = eval(f.read())
    
    img_name_list= sorted(list(idx_to_name[k] for k in idx_to_name.keys()))
    color_maps = ['PuOr', 'PuOr_r', 'RdYlBu',  'Spectral',  'gist_rainbow_r',  'hsv', 'hsv_r', ]
    save_projections_as_mat(floc, idx_to_name, mat_oploc)
    generate_and_save_maps(sio.loadmat(mat_oploc), img_name_list, figure_opdir, color_maps)