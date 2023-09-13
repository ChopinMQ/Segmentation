#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 22:18:15 2020

@author: wangxuelin
"""

#Python
import numpy as np
import configparser
import tensorflow as tf
from matplotlib import pyplot as plt
#Keras
from keras.models import model_from_json
from keras.models import Model
#scikit learn
import sys
sys.path.insert(0, './lib/')
# help_functions.py
from Functions.help_functions import *
# extract_patches.py
from Functions.extract_patches import recompone
from Functions.extract_patches import recompone_overlap
from Functions.extract_patches import get_data_testing
from Functions.extract_patches import get_data_testing_overlap
# pre_processing.py
from Functions.pre_processing import my_PreProc
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def Prediction():
    #========GPU Setting=========




    #========= CONFIG FILE TO READ FROM =======
    config = configparser.RawConfigParser()
    config.read("Vessel-Tech-master/configuration.txt")
    #===========================================
    #run the training on invariant or local
    path_data = config.get('data paths', 'path_local')
    # change the path_local

    #original test images 
    DRIVE_test_imgs_original = path_data + config.get('data paths', 'test_imgs_original')
    test_imgs_orig = load_hdf5(DRIVE_test_imgs_original)
    full_img_height = test_imgs_orig.shape[2]
    full_img_width = test_imgs_orig.shape[3]
    patch_height = int(config.get('data attributes', 'patch_height'))
    patch_width = int(config.get('data attributes', 'patch_width'))
    stride_height = int(config.get('testing settings', 'stride_height'))
    stride_width = int(config.get('testing settings', 'stride_width'))
    assert (stride_height < patch_height and stride_width < patch_width)
    #model name
    name_experiment = config.get('experiment name', 'name')
    path_experiment = 'Vessel-Tech-master/data/' +name_experiment +'/'
    #N full images to be predicted
    Imgs_to_test = int(config.get('testing settings', 'full_images_to_test'))
    #Grouping of the predicted images
    N_visual = int(config.get('testing settings', 'N_group_visual'))
    #====== average mode ===========
    average_mode = config.getboolean('testing settings', 'average_mode')

    #============ Load the data and divide in patches
    patches_imgs_test = None
    new_height = None
    new_width = None
    masks_test  = None
    if average_mode == True:
        patches_imgs_test, new_height, new_width = get_data_testing_overlap(
            DRIVE_test_imgs_original = DRIVE_test_imgs_original,  #original
            Imgs_to_test = int(config.get('testing settings', 'full_images_to_test')),
            patch_height = patch_height,
            patch_width = patch_width,
            stride_height = stride_height,
            stride_width = stride_width
        )
    else:
        patches_imgs_test = get_data_testing(
            DRIVE_test_imgs_original = DRIVE_test_imgs_original,  #original
            Imgs_to_test = int(config.get('testing settings', 'full_images_to_test')),
            patch_height = patch_height,
            patch_width = patch_width,
        )



    #================ Run the prediction of the patches ==================================
    best_last = config.get('testing settings', 'best_last')
    #Load the saved model
    model = model_from_json(open(path_experiment+name_experiment +'_architecture.json').read())
    model.load_weights(path_experiment+name_experiment + '_'+'best'+'_weights.h5')
    #Calculate the predictions

    # trying to allocate 7.85GiB
    print(patches_imgs_test.shape)
    predictions = model.predict(patches_imgs_test[0:25992], batch_size=32, verbose=2)
    print ("predicted images size :")
    print (predictions.shape)

    #===== Convert the prediction arrays in corresponding images
    pred_patches = pred_to_imgs(predictions, patch_height, patch_width, "original")

    print("###########################Finish predcting###########################")

    #========== Elaborate and visualize the predicted images ====================
    pred_imgs = None
    orig_imgs = None
    gtruth_masks = None


    if average_mode == True:
        pred_imgs = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)# predictions
        orig_imgs = my_PreProc(test_imgs_orig[0:pred_imgs.shape[0],:,:,:])    #originals
    else:
        pred_imgs = recompone(pred_patches,13,12)       # predictions
        orig_imgs = recompone(patches_imgs_test,13,12)  # originals




    orig_imgs = orig_imgs[:,:,0:full_img_height,0:full_img_width]
    pred_imgs = pred_imgs[:,:,0:full_img_height,0:full_img_width]
    print("Orig imgs shape: " +str(orig_imgs.shape))
    print("pred imgs shape: " +str(pred_imgs.shape))
    print(orig_imgs.shape, N_visual)
    visualize(group_images(orig_imgs,N_visual),"static/I/"+"all_originals")#.show()
    visualize(group_images(pred_imgs,N_visual),"static/I/"+"all_predictions")#.show()
    return 0
