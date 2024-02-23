# EJNMMI - DPNN - 20190120 21:08

## 20190121 23:58

Starting point of the `pretraining_phase_1` branch of the EJNMMI-DPNN project.

Using `21SIG_PXNET_PRETRAIN_SMOOTH_INPUT_CU_LEARNING` from `dpnn-archive` as base.

## 20190122 23:54

Added `run_exp_n_1_stage_1_min_max_norm_all_folds.py` from `dev1_hfx`'s `run_exp_n_1_stage_1_min_max_norm_all_folds_hfx.py`

## 20190123 02:15

added ..test_old.. files for checking structure of old dataset.

## 20190126 13:16
Keeping new files related to data-generator/dataset. Preparing to merge old_finetuning_experiment_22_GAP_sigmoid.

## 20190126 14:34
### Dev DATA stats
----------
New dev x specifications:
<class 'numpy.ndarray'>
(246, 95, 69, 79, 1)
float32
Mean 8041.11
Median 6372.97
Max 92152.0
Min 0.0
std.dev. 7255.93
----------
----------
dev y specifications:
<class 'numpy.ndarray'>
(246, 3, 1)
float32
Mean 0.333333
Median 0.0
Max 1.0
Min 0.0
std.dev. 0.471405
----------
test Data stats
----------
New test dataset specifications:
<class 'numpy.ndarray'>
(63, 95, 69, 79, 1)
float32
Mean 8040.48
Median 6370.58
Max 41896.2
Min 0.0
std.dev. 7025.66
----------
pretraining data stats
----------
dev x-tf specifications:
<class 'numpy.ndarray'>
(1077, 95, 69, 79, 1)
float32
Mean 10282.7
Median 7646.11
Max 95617.8
Min -3173.77
std.dev. 9765.27
----------
----------
dev y-tf specifications:
<class 'numpy.ndarray'>
(1077, 95, 69, 1)
float32
Mean 4.99517
Median 5.86557
Max 11.3836
Min -0.138251
std.dev. 2.73669
----------

# 20190126 15:40

Copied models from exp_n_1s2 for selection.
Using all epoch@240

# 20190126 20:28
Need to make some changes in evaluation script.
Model with lowest loss(e.g. from fold 4 & 5) had low TPR -- may try other models.
Will run the finetuning again - with different seed. 

# 20190210 16:49 IST
- Will be running expn_n_3 - SET_8 (with $f^{-2}$ as weights.)

# 20190220 23:34 IST
- Changing run_exp_n_3_pxnet_GAP_sigmoid_xavier.py -> run_exp_n_5_pxnet_GAP_sigmoid_xavier_no_pretrain.py
- Also some changes will be made in the model of PXNET_SIGMOID_GAP.py 

