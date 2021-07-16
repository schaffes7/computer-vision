# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:46:11 2020

@author: Not Your Computer
"""

import os
import pandas as pd
import cv2
import time

import numpy as np
#np.random.seed(1)

import matplotlib.pyplot as plt

import random
from random import shuffle
#random.seed(1)

from skimage.transform import resize

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import concatenate, Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, UpSampling2D, Input, ZeroPadding2D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split

from CVUtils import LoadImages, CalcDispersion, LoadBackgrounds, MakePFrame, GetImagePaths, GenerateAnchors, GetTruthPaths, LoadTruthImages, FindCentroids, TimeTag

print('\n[CV_SegmentationModel]: ALL LIBRARIES LOADED.')

#%%
#==============================================================================
#                              MAIN SCRIPT
#==============================================================================

# Directories
folders = ['D:\\Images\\Logan',
           'D:\\Images\\Morgan',
           'D:\\Images\\TheGirls']
bg_folders = ['D:\\Images\\NegativeScenes']

# Settings
binary = True
model_dir = 'D:\\CVFiles\\ANNIE\\Models'

# Image Parameters
n_imgs = 200; n_backgrounds = 50
#n_imgs = 50; n_backgrounds = 50

frame_h = 480; frame_w = 640; n_channels = 3
s = 480

classes = []
for folder in folders:
    classes.append(folder.split('\\')[-1].lower())

if binary:
    frame_size = (frame_h,frame_w)
    n_classes = 1
    outfile = '{}\\segmentor_binary.h5'.format(model_dir)
else:
    frame_size = (s,s)
    n_classes = len(folders)
    outfile = '{}\\segmentor_{}.h5'.format(model_dir, s)

# Initialize Variables
class_dict = {}; best_val_metric = 0

#%%
#==============================================================================
#                                BUILD MODEL
#==============================================================================
train_model = True
n_models = 1
baseline = 0.75
batch_size = 1
n_epochs = 1
patience = 5
test_ratio = 0.20

if train_model:
            
    # COLLECT IMAGE PATHS    
    img_paths, truth_paths = GetTruthPaths(folders, shuffle_paths = True)
    if len(img_paths) < n_imgs:
        n_imgs = len(img_paths)
    img_paths = img_paths[0:n_imgs]; truth_paths = truth_paths[0:n_imgs]
    
    if len(bg_folders) > 0:
        bg_img_paths, bg_truth_paths = GetTruthPaths(bg_folders, shuffle_paths = True)
        if len(bg_img_paths) < n_backgrounds:
            n_backgrounds = len(bg_img_paths)
        img_paths += bg_img_paths[0:n_backgrounds]
        truth_paths += bg_truth_paths[0:n_backgrounds]

    # SPLIT IMAGE PATHS
    x_train, x_test, y_train, y_test = train_test_split(img_paths, truth_paths, test_size = test_ratio, shuffle = True)
    
    print('\n[CV_SegmentationModel]: LOADING IMAGES...')
    if binary:
        x_train = LoadImages(x_train, s = (frame_h, frame_w))
        x_test = LoadImages(x_test, s = (frame_h, frame_w))
    else:
        x_train = LoadImages(x_train, s = (s,s))
        x_test = LoadImages(x_test, s = (s,s))
    print('[CV_SegmentationModel]: IMAGES LOADED.')
    
    print('\n[CV_SegmentationModel]: LOADING MASKS...')
    if binary:
        y_train = LoadTruthImages(y_train, binary = binary, bg_folders = bg_folders, s = (frame_h, frame_w))
        y_test = LoadTruthImages(y_test, binary = binary, bg_folders = bg_folders, s = (frame_h, frame_w))
    else:
        y_train = LoadTruthImages(y_train, binary = binary, bg_folders = bg_folders, s = (s,s))
        y_test = LoadTruthImages(y_test, binary = binary, bg_folders = bg_folders, s = (s,s))
    print('[CV_SegmentationModel]: MASKS LOADED.')
    
    print('\n[CV_SegmentationModel]: IMAGE COUNT:   {}'.format(len(x_train)))
    print('[CV_SegmentationModel]: ---------------------------')
    print('[CV_SegmentationModel]: Training:      {}'.format(len(y_train)))
    print('[CV_SegmentationModel]: Test:          {}'.format(len(y_test)))
    print('')

    
    for I in range(n_models):
        # HYPERPARAMETERS
        if binary:
            metric = 'binary_accuracy'
            val_metric = 'val_binary_accuracy'
            opt_type = 'adam'
            losses = random.choice(['mean_squared_error','mean_absolute_error'])
            final_act = 'sigmoid'
        else:
            metric = 'categorical_accuracy'
            val_metric = 'val_categorical_accuracy'
            opt_type = 'adam'
            losses = 'categorical_crossentropy'
            final_act = 'softmax'
            
        learn_rate = random.choice([5e-8, 1e-8, 5e-9, 1e-9])
        dropout = random.choice([0.0, 0.20, 0.33, 0.50])
        pool_size = random.choice([(2,2)])
        final_pool = random.choice([(1,1),(3,3)])
        
        print('\nMODEL {} SUMMARY: '.format(I))
        print('---------------------------------------')
        print('Learn Rate:       ', learn_rate)
        print('Loss Type:        ', losses)
        print('Optimizer:        ', opt_type)
        print('Pool Size:        ', pool_size)
        print('Dropout:          ', dropout)
        print('Final Activation: ', final_act)
        print('---------------------------------------')
        print()
            
#        # INPUT
#        if binary: img_input = Input(shape = (frame_h, frame_w, 3))
#        else: img_input = Input(shape = (s, s, 3))
#        # CONV
#        conv = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(img_input)
#        conv = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(conv)
#        conv = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(conv)
#        conv = BatchNormalization()(conv)
#        conv1 = Dropout(dropout)(conv)
#        print('Conv: ', list(conv1.get_shape()))
#        # POOL
#        pool1 = MaxPooling2D(pool_size)(conv1)
#        print('Pool: ', list(pool1.get_shape()))
#        # CONV
#        conv = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(pool1)
#        conv = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(conv)
#        conv = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(conv)
#        conv = BatchNormalization()(conv)
#        conv2 = Dropout(dropout)(conv)
#        print('Conv: ', list(conv2.get_shape()))
#        # POOL
#        pool2 = MaxPooling2D(pool_size)(conv2)
#        print('Pool: ', list(pool2.get_shape()))
#        # CONV
#        conv = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(pool2)
#        conv = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(conv)
#        conv = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(conv)
#        conv = BatchNormalization()(conv)
#        conv3 = Dropout(dropout)(conv)
#        print('Conv: ', list(conv3.get_shape()))
#        # UNPOOL
#        up1 = concatenate([UpSampling2D(pool_size)(conv3), conv2])
#        print('Up2D: ', list(up1.get_shape()))
#        # CONV
#        conv = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(up1)
#        conv = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(conv)
#        conv = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(conv)
#        conv = BatchNormalization()(conv)
#        conv4 = Dropout(dropout)(conv)
#        print('Conv: ', list(conv4.get_shape()))
#        # UNPOOL
#        up2 = concatenate([UpSampling2D(pool_size)(conv4), conv1])
#        print('Up2D: ', list(up2.get_shape()))
#        # CONV
#        conv = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(up2)
#        conv = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(conv)
#        conv = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(conv)
#        conv = BatchNormalization()(conv)
#        conv5 = Dropout(dropout)(conv)
#        print('Conv: ', list(conv5.get_shape()))
#        out = Conv2D(n_classes, final_pool, activation = final_act, padding = 'same')(conv5)
#        print('Conv: ', list(out.get_shape()))
#        model = Model(img_input, out)
            
            
        # INPUT
        if binary: img_input = Input(shape = (frame_h, frame_w, 3))
        else: img_input = Input(shape = (s, s, 3))
        
        # CONV 1
        conv = Conv2D(16, (1,1), activation = 'relu', padding = 'same')(img_input)
        conv = Conv2D(16, (3,3), activation = 'relu', padding = 'same')(conv)
        conv = BatchNormalization()(conv)
        conv1 = Dropout(dropout)(conv)
        print('Conv 1: ', list(conv1.get_shape()))
        
        # POOL 1
        pool1 = MaxPooling2D(pool_size)(conv1)
        print('Pool 1: ', list(pool1.get_shape()))
        
        # CONV 2
        conv = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(pool1)
        conv = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(conv)
        conv = BatchNormalization()(conv)
        conv2 = Dropout(dropout)(conv)
        print('Conv 2: ', list(conv2.get_shape()))
        
        # POOL 2
        pool2 = MaxPooling2D(pool_size)(conv2)
        print('Pool 2: ', list(pool2.get_shape()))
        
        # CONV 3
        conv = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(pool2)
        conv = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(conv)
        conv = BatchNormalization()(conv)
        conv3 = Dropout(dropout)(conv)
        print('Conv 3: ', list(conv3.get_shape()))
        
        # POOL 3
        pool3 = MaxPooling2D(pool_size)(conv3)
        print('Pool 3: ', list(pool3.get_shape()))
        
        # CONV 4
        conv = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(pool3)
        conv = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(conv)
        conv = BatchNormalization()(conv)
        conv4 = Dropout(dropout)(conv)
        print('Conv 4: ', list(conv4.get_shape()))
        
        # UNPOOL 1
        up1 = concatenate([UpSampling2D(pool_size)(conv3), conv2])
        print('Up2D 1: ', list(up1.get_shape()))
        
        # CONV 5
        conv = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(up1)
        conv = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(conv)
        conv = BatchNormalization()(conv)
        conv5 = Dropout(dropout)(conv)
        print('Conv 5: ', list(conv5.get_shape()))
        
        # UNPOOL 2
        up2 = concatenate([UpSampling2D(pool_size)(conv4), conv1])
        print('Up2D 2: ', list(up2.get_shape()))
        
        # CONV 6
        conv = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(up2)
        conv = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(conv)
        conv = BatchNormalization()(conv)
        conv6 = Dropout(dropout)(conv)
        print('Conv 6: ', list(conv6.get_shape()))
        
        # UNPOOL 3
        up3 = concatenate([UpSampling2D(pool_size)(conv6), conv2])
        print('Up2D 3: ', list(up3.get_shape()))
        
        # CONV 7
        conv = Conv2D(16, (3,3), activation = 'relu', padding = 'same')(up3)
        conv = Conv2D(16, (3,3), activation = 'relu', padding = 'same')(conv)
        conv = BatchNormalization()(conv)
        conv7 = Dropout(dropout)(conv)
        print('Conv 7: ', list(conv7.get_shape()))
        
        # OUT CONV
        out = Conv2D(n_classes, final_pool, activation = final_act, padding = 'same')(conv5)
        print('Conv Out: ', list(out.get_shape()))
        
        model = Model(img_input, out)        
        print('\n---------------------------------------')
        
        # DEFINE OPTIMIZER
        if opt_type == 'rms':
            opt = keras.optimizers.RMSprop(lr = learn_rate, rho = 0.9, epsilon = None, decay = 1e-6)
        if opt_type == 'sgd':
            opt = keras.optimizers.SGD(lr = learn_rate, decay = 1e-6)
        if opt_type == 'adam':
            opt = keras.optimizers.Adam(lr = learn_rate, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = 0.0, amsgrad = False)
        if opt_type == 'adagrad':
            opt = keras.optimizers.Adagrad(lr = learn_rate, epsilon = None, decay = 0.0)
        if opt_type == 'adadelta':
            opt = keras.optimizers.Adadelta(lr = learn_rate, rho = 0.95, epsilon = None, decay = 0.0)
        
        # COMPILE MODEL
        if binary: model.compile(loss = losses, optimizer = opt, metrics = [metric])
        else: model.compile(loss = losses, optimizer = opt, metrics = [metric])
        
        es = EarlyStopping(monitor = val_metric, mode = 'max', verbose = 1, patience = patience, baseline = baseline)
    #        mc = ModelCheckpoint(outfile, monitor = val_metric, mode = 'max', verbose = 1, save_best_only = True)
    
        # TRAIN MODEL
        histories = []
        print('\n[CV_SegmentationModel]: TRAINING MODEL...')
        histories.append(model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = n_epochs, batch_size = batch_size, callbacks = [es]))
        
        # PLOT HISTORY
        if n_epochs > 1 and patience > 1:
            plt.plot(histories[0].history[metric])
            plt.plot(histories[0].history[val_metric])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
                
        # SAVE MODEL
        final_metric = histories[0].history[metric][-1]
        final_val_metric = histories[0].history[val_metric][-1]
        if final_val_metric > best_val_metric and final_val_metric > baseline:
            if final_val_metric > final_metric:
                print('\n[CV_SegmentationModel]: SAVING MODEL...', outfile)
                model.save(outfile)
                print('[CV_SegmentationModel]: MODEL SAVED TO: {}'.format(outfile))
                best_val_metric = final_val_metric
                print('\n[CV_SegmentationModel]: NEW BEST VAL_ACCURACY: ', best_val_metric)
    else:
        
        # LOAD SAVED MODEL
        if binary: segment_path = r"D:\CVFiles\ANNIE\Models\segmentor_binary.h5"
        else: segment_path = r"D:\CVFiles\ANNIE\Models\segmentor_480.h5"
        model = load_model(segment_path)
    
    # PREVIEW SEVERAL TEST PREDICTIONS
    n = 6; fs = (8,6)
    for i in range(n):
        p = model.predict(x_test[i:i+1])[0]
        plt.figure(figsize = fs)
        plt.imshow(p[:,:,0])
        plt.show()
        


#%%
# Segmentor Settings
p_thresh = 0.75

# Video Settings
run_camera = True
font = cv2.FONT_HERSHEY_DUPLEX
stop_camera_key = 'q'
alpha = 0.50
anchors = GenerateAnchors(frame_h, frame_w, s, search_scales = [1], overlap = 0)
class_colors = [(40,40,225),(40,225,40),(225,40,40),(50,50,210),(170,210,80),(180,200,40),(180,225,40),(180,225,40)]
hud_color = (220,220,50)

if run_camera:
    
    rad_list = []; fps = 0; frame_no = 0
    
    # CREATE VIDEO CAPTURE OBJECT
    cap = cv2.VideoCapture(0)
    
    st = time.time()
    
    while(True):
        # CAPTURE FRAME
        ret, frame = cap.read()
        output = frame.copy()
        frame = np.array(frame)

        # SEGMENT FRAME
        if binary: layers = model.predict(np.reshape(frame, (1, frame_h, frame_w, n_channels)))
        else: layers = model.predict(np.reshape(frame[:,80:frame_w-80,:], (1, s, s, n_channels)))
        
        L = 0
        lyr_out = np.zeros([frame_h, frame_w, n_channels], dtype = np.uint8)
        
        for lidx in range(np.shape(layers)[-1]):
            lyr = layers[0][:,:,lidx]
            layer_color = class_colors[lidx]
            hot_idx = np.where(lyr >= p_thresh)
            
            if binary: offset = 0
            else: offset = 80
            
            lyr_out[hot_idx[0], hot_idx[1]+offset, 0] = layer_color[0]
            lyr_out[hot_idx[0], hot_idx[1]+offset, 1] = layer_color[1]
            lyr_out[hot_idx[0], hot_idx[1]+offset, 2] = layer_color[2]
#            lyr_out[0:10,:,:] = 0; lyr_out[(frame_h - 10):frame_h,:,:] = 0
#            lyr_out[:,470:,:] = 0; lyr_out[:,0:90,:] = 0
            if binary: cv2.putText(output, 'Positive', (580, 17*(L+1)), font, 0.35, class_colors[lidx], 1, cv2.LINE_AA)
            else: cv2.putText(output, classes[lidx], (580, 17*(L+1)), font, 0.35, class_colors[lidx], 1, cv2.LINE_AA)

            box_w = 60; box_h = 15; buffer = 3
            x1 = 575; y1 = 7 + (box_h + buffer)*(L)
            x2 = x1 + box_w; y2 = y1 + box_h
            cv2.rectangle(output, (x1,y1), (x2,y2), layer_color, 2)
            L += 1
            
        # ADD OVERLAY
        if frame_no % 7 == 0:
            dt = time.time() - st
            fps = round(7 / dt)
            st = time.time()
        
        output = cv2.addWeighted(lyr_out, alpha, output, 1-alpha, 0)
        
        cv2.putText(output, 'binary:   {}'.format(binary), (5, 15), font, 0.40, hud_color, 1, cv2.LINE_AA)
        cv2.putText(output, 'p_thresh: {}'.format(p_thresh), (5, 30), font, 0.40, hud_color, 1, cv2.LINE_AA)
        cv2.putText(output, 'alpha:    {}'.format(alpha), (5, 45), font, 0.40, hud_color, 1, cv2.LINE_AA)
        cv2.putText(output, 'FPS:      {}'.format(fps), (5, 60), font, 0.40, hud_color, 1, cv2.LINE_AA)

        # DISPLAY FRAME
        cv2.imshow('frame', output)
    
        if cv2.waitKey(1) & 0xFF == ord(stop_camera_key):
            break
            
        frame_no += 1
    
    # DISCONNECT CAMERA FEED
    cap.release()
    cv2.destroyAllWindows()