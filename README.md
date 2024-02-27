# nnUnet_OnWindows
To run nnUnet on Windows.
The original code is recommended to run on a Linux system, and this code can be run directly on a Windows system.
I am still updating, if there is a problem, I will try my best to solve it :)

## introduction
For more information about nnU-Net, please read the following paper:

Isensee, Fabian, et al. "nnU-Net: Breaking the Spell on Successful Medical Image Segmentation." arXiv preprint arXiv:1904.08128 (2019).

The code is adapted from:

https://github.com/MIC-DKFZ/nnUNet

Thank u~

## Usage

In order to facilitate debugging, the command line cannot be used directly.

### 1.Run paths.py

Change you paths and it can create a folder names "Data".

### 2.Run file in "dataset_conversion"

nnU-Net expects datasets in a structured format. 

You should move splits_final.pkl to nnUNet_preprocessed fold for training.

### 3.Run nnUNet_plan_and_preprocess.py in "experiment_planning"

To create subfolders with preprocessed data for the 2D U-Net as well as all applicable 3D U-Nets.

### 4.Run run_training.py in "run"

To start training.

### 5.Run predict_simple.py in "inference"

## Issue
1. RuntimeError: CUDA out of memory. 

Solution: Adjust the batch_size yourself in nnunet/experience_planning/change_batch_size.py



