# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:01:26 2020

@author: Not Your Computer
"""

# LOAD MODULES
import os
import time
import pandas as pd
import numpy as np
import random
import cv2
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications import imagenet_utils, MobileNet
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageFilter, ImageEnhance
from CVUtils import Segmentor, Detector, Classifier
from CVUtils import GenerateAnchors
import matplotlib.pyplot as plt
# SET RANDOM SEEDS
random.seed(1)
np.random.seed(1)
print('\nALL MODULES LOADED.')

#%%
# GENERATE N RANDOM COLORS
def Colors(n_colors):
    colors = []
    for i in range(n_colors):
        rgb = [random.randint(40,220),random.randint(100,220),random.randint(200,220)]
        random.shuffle(rgb)
        colors.append(tuple(rgb))
    return colors


def FillMap(lyr, rad = 3):
    mod_lyr = np.zeros(lyr.shape)
    for i in range(lyr.shape[0]):
        for j in range(lyr.shape[1]):
            i_st = max(0, i-rad)
            i_end = min(lyr.shape[0], i+rad)
            j_st = max(0, j-rad)
            j_end = min(lyr.shape[1], j+rad)
            mod_lyr[i,j] = round(np.mean(lyr[i_st:i_end, j_st:j_end]))
    return mod_lyr


def DrawClassDetails(output, params, class_text = True, bar_chart = False):
    i = 0
    for val in pc:
        class_str = '{}% : {}'.format(int(100*val), classes[i])
        if class_text: cv2.putText(output, class_str, (C + spread, R - spread + 20*i), params['font'], 0.50, class_colors[i], 1, params['line_style'])
        if bar_chart: cv2.rectangle(output, (C + spread + 30, R - spread + 20*i - 20), (C + spread + 30 + int(100*val), R - spread + 20*i), class_colors[i], thickness = -1)
        i += 1
        
        
def DrawHUD(output, params):
    hud_labs = ['Size:         {}'.format(params['s']),
                'DetThresh:    {}'.format(params['detect_threshold']),
                'Alpha:        {}'.format(params['alpha']),
                'MinPixels:    {}'.format(params['min_pixels']),
                'FPS:          {} Hz'.format(params['fps'])]
    J = 0
    for lab in hud_labs:
        cv2.putText(output, lab, (5, 15 + J*15), params['font'], 0.40, params['hud_color'], 1, params['line_style'])
        J += 1

#%%
# PARAMETERS
params = {'detector_path':r"D:\CVFiles\ANNIE\Models\segmentor_binary_demo.h5",
          'classifier_path':r"D:\CVFiles\ANNIE\Models\classifier_32.h5",
          'min_pixels':1000,
          'frame_width':640,
          'frame_height':480,
          'n_channels':3,
          'detect_threshold':0.95,
          'font':cv2.FONT_HERSHEY_DUPLEX,
          'stop_camera_key':'q',
          'alpha':0.50,
          'fps_averaging':5,
          'hud_color':(220,220,50),
          'anchor_overlap':0,
          'search_scales':[2,3,4,6],
          's':32,
          'line_style':cv2.LINE_AA,
          'fps':0,
          'classes':[],
          'class_colors':[]
          }


# GENERATE FRAME ANCHORS
anchors = GenerateAnchors(params['frame_height'], params['frame_width'], params['s'], search_scales = params['search_scales'], overlap = params['anchor_overlap'])


# READ MODEL PARAMETERS FILE
model_params_path = "D:\\CVFiles\\ANNIE\\Models\\classes.txt".format(os.path.basename(params['classifier_path']))
with open(model_params_path, 'r') as f:
    classes = f.read()
classes = classes.split(',')
params['classes'] = classes


# GENERATE CLASS COLORS
class_colors = Colors(len(classes))
params['class_colors'] = class_colors


# IMPORT MODELS
DET = Segmentor(model_path = params['detector_path'], thresh = params['detect_threshold'])
CLS = Classifier(model_path = params['classifier_path'], classes_path = model_params_path)


#%%
histories = {}; C_histories = {}; R_histories = {}
for i in range(len(classes)):
    histories[i] = list(np.zeros(params['fps_averaging']))
    R_histories[i] = list(np.zeros(params['fps_averaging']))
    C_histories[i] = list(np.zeros(params['fps_averaging']))

# CREATE VIDEO CAPTURE OBJECT
cap = cv2.VideoCapture(0)
st = time.time()
frame_no = 0
fps = 0

# MAIN LOOP
while(True):
    # CAPTURE FRAME
    ret, frame = cap.read()
    output = frame.copy()
    frame = np.array(frame)
    
    
    # SEGMENT FRAME
    layers = DET.predict(np.reshape(frame, (1, params['frame_height'], params['frame_width'], params['n_channels'])))
    L = 0; n_layers = 1
    for lidx in range(n_layers):
        lyr = layers[lidx]
        
        # DENOISE LAYER
#        lyr = cv2.fastNlMeansDenoising(layers[lidx], h = 3)
        
        # MAKE DETECTIONS (currently assumes 1 at most per layer)
        [centroid, spread] = Detector(lyr, method = 'mean', thresh = params['detect_threshold'], min_pixels = params['min_pixels'])
        layer_color = class_colors[L]
        
        hot_idx = np.where(lyr > 0)
        lyr_out = np.zeros([params['frame_height'], params['frame_width'], params['n_channels']], dtype = np.uint8)
        lyr_out[hot_idx[0], hot_idx[1], 0] = layer_color[0]
        lyr_out[hot_idx[0], hot_idx[1], 1] = layer_color[1]
        lyr_out[hot_idx[0], hot_idx[1], 2] = layer_color[2]
        
        # ADD SEGMENTATION OVERLAY
        output = cv2.add(lyr_out, output)

        # IF NO DETECTS, SKIP TO NEXT FRAME
        if centroid == [0,0] and spread == 0:
            pass
        
        # ELSE CLASSIFY DETECTS
        else:
            # GENERATE SUBFRAME
            R = centroid[0]; C = centroid[1]
            x1 = C - spread; x2 = C + spread
            y1 = R - spread; y2 = R + spread
            sf = frame[y1:y2, x1:x2, :]
            sf = np.resize(np.array(sf), (1, params['s'], params['s'], params['n_channels']))
            
            # CLASSIFY DETECTS
            pc = CLS.predict(sf)
            cidx = np.argmax(pc)
            cname = classes[cidx]
            c_color = class_colors[cidx]
            
            # DRAW CLASS DETAILS
            DrawClassDetails(output, params, class_text = True, bar_chart = False)
            
            # DRAW CIRCLE ON FRAME
            rad_list = histories[L]
            rad_list = rad_list[1:] + [spread]
            rad = int(round(np.mean(rad_list)))
            cv2.circle(output, (C,R), rad,  c_color, 2)
            
            # UPDATE HISTORIES
            histories[L] = rad_list
            
        L += 1
    
    
    # CALCULATE FPS
    if frame_no % params['fps_averaging'] == 0:
        params['fps'] = round(params['fps_averaging'] / (time.time() - st))
        st = time.time()
        

    # DRAW HUD OVERLAY
    DrawHUD(output, params)
    
    
    # SHOW FRAME
    cv2.imshow('frame', output)
    if cv2.waitKey(1) & 0xFF == ord(params['stop_camera_key']):
        break
    
    frame_no += 1

# DISCONNECT CAMERA FEED
cap.release()
print('\nCAMERA CONNECTION RELEASED.')
cv2.destroyAllWindows()

print('\nDONE.')