# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 20:40:45 2019

@author: Not Your Computer
"""

import os
import time
import pandas as pd
import numpy as np
from numpy.random import seed
import random; from random import shuffle
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy import interp; from scipy.misc import imsave
from itertools import cycle
from tensorflow import set_random_seed
from skimage.transform import resize
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import skimage; from skimage import transform; from skimage.transform import resize
import cv2
import h5py
from random import shuffle
from skimage.transform import resize
from matplotlib import pyplot as plt
import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)
random.seed(1)
np.random.seed(1)
from PIL import Image, ImageFilter, ImageEnhance
from PIL.ImageFilter import GaussianBlur
from scipy.misc import imsave
from CVUtils import timeTag, CalcAnchors, rgb2gray, SplitFrame, CNN, Intersection, Union, IOU, processImage, CalcBoundaries

#%%

# SET PARAMS
targets = ['morgan','logan']
test_ratio = 0.10
valid_ratio = 0.10

df = pd.read_csv('D:\\Images\\BBoxes.csv')
df = df[df['cat'].isin(targets)]
img_paths = list(df['fpath'])
classes = df['cat']
i = 0
for i in range(len(targets)):
    classes[classes == targets[i]] = i
    i += 1
classes = list(classes)
classes = to_categorical(classes)

# CALCULATE & SAVE ANCHOR BOX BOUNDARIES
anchor_df = CalcBoundaries(480, 640, 80, search_scales = [1], overlap = None,
                           black_bar = 0, outfile = 'D:\\Images\\anchors.csv')

# CREATE TRAIN / TEST / VALID SPLITS
print('\n[CV_TrainCNN]: PARTITIONING IMAGE PATHS...')
x_train, x_valid, y_train, y_valid = train_test_split(img_paths, classes, test_size = test_ratio, shuffle = True)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = valid_ratio, shuffle = True)
print(np.unique(y_train), np.unique(y_test), np.unique(y_valid))
print('[CV_TrainCNN]: ---------------------------')
print('[CV_TrainCNN]: Training:      {}'.format(len(y_train)))
print('[CV_TrainCNN]: Test:          {}'.format(len(y_test)))
print('[CV_TrainCNN]: Validation:    {}'.format(len(y_valid)))

#%%

# PRODUCE TRAINING SAMPLES
x_train_imgs = []
y_train_classes = []
for i in range(len(x_train)):
    sf_stack, sf_classes = processImage(x_train[i], anchor_df)
    x_train_imgs += sf_stack
    y_train_classes += sf_classes
x_train_imgs = np.array(x_train_imgs)
y_train_classes = np.array(y_train_classes)

# PRODUCE TEST SAMPLES
x_test_imgs = []
y_test_classes = []
for i in range(len(x_test)):
    sf_stack, sf_classes = processImage(x_test[i], anchor_df)
    x_test_imgs += sf_stack
    y_test_classes += sf_classes
x_test_imgs = np.array(x_test_imgs)
y_test_classes = np.array(y_test_classes)

# PRODUCE VALIDATION SAMPLES
x_valid_imgs = []
y_valid_classes = []
for i in range(len(x_valid)):
    sf_stack, sf_classes = processImage(x_valid[i], anchor_df)
    x_valid_imgs += sf_stack
    y_valid_classes += sf_classes
x_valid_imgs = np.array(x_valid_imgs)
y_valid_classes = np.array(y_valid_classes)

#%%

final_act = 'relu'; learn_rate = 0.01; opt_type = 'rms'
losses = 'categorical_crossentropy'

# INITIALIZE MODEL
model = Sequential()
model.add(Conv2D(12, (3,3), input_shape = (80,80,3), activation = 'relu', data_format = 'channels_last'))
model.add(MaxPooling2D(pool_size = (2,2)))              # 64 ==> 32
model.add(Dropout(0.25))                                # DROP

model.add(Conv2D(24, (3,3), activation = 'relu'))       # CONV 2D
model.add(Conv2D(24, (3,3), activation = 'relu'))       # CONV 2D
model.add(MaxPooling2D(pool_size = (2,2)))              # 32 ==> 16
model.add(Dropout(0.25))                                # DROP

model.add(Conv2D(32, (3,3), activation = 'relu'))       # CONV 2D
model.add(Conv2D(32, (3,3), activation = 'relu'))       # CONV 2D
model.add(MaxPooling2D(pool_size = (2,2)))              # 16 ==> 8
model.add(Dropout(0.25))                                # DROP

model.add(Flatten())                                    # FLATTEN
                             
model.add(Dense(1024, activation = 'relu'))             # DENSE
model.add(Dropout(0.25))                                # DROP

model.add(Dense(128, activation = 'relu'))              # DENSE
model.add(Dropout(0.25))                                # DROP

model.add(Dense(len(targets), activation = final_act))  # DENSE

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
model.compile(loss = losses, optimizer = opt, metrics = ['categorical_accuracy'])
print('\n[CV_TrainCNN]: MODEL COMPILED.')

# TRAIN MODEL
print('\n[CV_TrainCNN]: TRAINING MODEL...')
model.fit(x_train_imgs, y_train_classes, verbose = 1, epochs = 1, validation_data = [x_test_imgs, y_test_classes])
print('\n[CV_TrainCNN]: MODEL TRAINED.')

# EVALUATE MODEL (VALIDATION)
print('\n[CV_TrainCNN]: EVALUATING MODEL...')
model.evaluate(x_valid_imgs, y_valid_classes)
print('\n[CV_TrainCNN]: DONE.')
