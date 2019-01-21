===================================================
THING ADDED SO FAR:
  >>> run.py
  >>> AgeRegNet.py: Cost function has not been removed from the model yet
  >>> data.py: has a simple load function (No generator has been added)
  >>> train.py: Trainer class
  >>> utility.py: helper functions for folder management etc.
  >>> augmentation.py: gaussian+translation method (OPTIMIZATION REQUIRED)

===================================================
Date: 2017-20-12

Date: 2017-21-12

Date: 2017-22-12

Date: 2017-23-12
  >>> Added _projection_net.py: It has almost been finished
      Need to test and add loss function only to complete it.
  SAVED AS: 2017_12_23_2244_Master_HUB_CODES.tar.gz



===================================================
Date: 2017-24-12 (SUNDAY)
  >>> Completed the implementation of the _projection_net an tested on small
    batch. However final testing might be required, and minor performance
    enhancements.
  >>> Trainer class will be split tomorrow for sure.
  SAVED AS: 2017_12_25_0126_Master_HUB_CODES.tar.gz

===================================================
Date: 2017-25-12 (CHRISTMAS DAY)
  >>> Basic frame of the _train.py has been made.

===================================================
Date: 2017-26-12 (TUESDAY)
  >>> logger.py has been completed. (FINE_TUNING based on compatibility with
      other modules is needed though.)
      SAVED AS: 2017_12_27_0118_Master_HUB_CODES.tar.gz
===================================================
Date: 2017-27-12

Date: 2017-28-12

Date: 2017-29-12

Date: 2017-30-12 (SATURDAY)
  >>> major development has been done in the train.py. train_dev_v2.py id the
      file with latest changes. It will be replace the train.py in next(or next
      few) versions.
      SAVED AS: 2017_12_29_2341_Master_HUB_Codes.tar.gz

Date: 2017-31-12

Date: 2018-01-01 -- 2018-05-01

Date: 2018-01-06 (SATURDAY)
  >>> Processed new parkinson data. Create _x_parkinson_data.py (NOTE: float64-
      float32 issue)
  >>> After creating this, functions in data.py added to use this data. First
      round check done for the DataGenerator functions.
      SAVED IN: 2018_01_07_0230_MASTER_HUB_CODES

Date: 2018-01-07

Date: 2018-01-08

Date: 2018-01-09

Date: 2018-01-10

Date: 2018-01-11

Date: 2018-01-12

Date: 2018-01-13

Date: 2018-01-14

Date: 2018-01-15

Date: 2018-01-16

Date: 2018-01-17 (WEDNESDAY)
  >>> Added _x_making_3d_mnist.py for making 3d mnist for performing checks.
  >>> Accuracy calculation setup was added in the train.py for some code sets.(Could be merged in new versions)
  
Date: 2018-01-18

Date: 2018-01-19

Date: 2018-01-20 (SATURDAY)
  >>> Will proceed the work on developing tester class/functions.
  >>> New functions to utility will be added
  >>> SAVED_AS: 2018_01_20_2126_Master_HUB_Codes
  
Date: 2018-01-20

Date: 2018-01-21

Date: 2018-01-22

Date: 2018-01-23

Date: 2018-01-24 (WEDNESDAY)
  >>> Added `Specialized` folder, for storing scripts that has been used and/or may be used in the future for some very specific tasks.
  >>> Added Specialized/score_calculator/score_calculator.py
  >>> Added Specialized/score_calculator/_x_final_scores_and_plots.py
  >>> Added Specialized/entry_log
  >>> Expected developments:
        * Add script for pre-training  the compression part of the projection net.
        * Saving and storing selected weights
        * Script for training the network after pre-training--- may be taking advantage of the new data.py file and its DataGenerator class.
        
===================================================

Date: 2018-01-25 (THURSDAY)
  >>> Added projection_net_avg_pooling_pretraining.py for pretraining on projection data.
        * Codes in DELTA and DZNE will be same from now onwards. (To avoid compatibility issues.)
        * Normal Trainer is not enough for executing pre-training and training seamlessly, therefore new run.py will be made using DataGenerator
  >>> Added projection_net_avg_pooling_pretraining.py for pretraining.
  >>> Added functions in cost_functions.py
  >>> Added two functions in utility.py
  >>> SAVED AS: 2018_01_25_2011_IST_Master_HUB_Codes

Date: 2018-01-26

Date: 2018-01-27

Date: 2018-01-28
==================================================
Date: 2018-01-29 (MONDAY)
  >>> Changed indexing method in DataGenerator of data.py for avoiding passing data by reference.(i.e. to give a copy of data rather than a reference)
  >>> logger.py modified and _x_testing_logger_py_... .py added for testing logger
  >>> SAVED AS: 2018_01_30_0009_IST_Master_HUB_Codes

==================================================
Date: 2018-01-30 (MONDAY)
  >>> Working prototype of logger.py made. Certaing enhancement and testing will be done.
  >>> _x_.. . py for testing logger added
  >>> SAVED AS: 2018_01_30_2153_IST_Master_HUB_Codes

Date: 2018-01-31 (TUESDAY)
  >>> logger.py semi-finalized.(Preliminary tests completed)
  >>> Some modifications in Net (of net.py) made.
  >>> _x_auto_encoder_proj_net.py added, to explore Autoencoders
  >>> SAVED AS: 2018_01_31_0052_IST_Master_HUB_Codes
  >>> logger.py tested and finalized
  >>> PretrainingPXNetFirstHalf.py , run_PretrainingPXNetFirstHalf.py and PretrainingPXNet.py added. These will be further edited and standardized.
  >>> SAVED AS: 2018_01_31_2311_IST_Master_HUB_Codes
