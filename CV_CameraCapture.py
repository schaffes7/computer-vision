# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 18:15:40 2019

@author: Not Your Computer
"""

import numpy as np
np.random.seed(1)

import random
random.seed(1)

import cv2

from CVUtils import TimeTag


# CREATE VIDEO CAPTURE OBJECT
font = cv2.FONT_HERSHEY_SIMPLEX
frame_no = 0
stop_camera_key = 'q'
folder = 'D:\\CameraCaptures'

cap = cv2.VideoCapture(0)
while(True):
    
    # CAPTURE FRAME
    ret, frame = cap.read()
    
    # CONVERT FRAME TO GRAYSCALE
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # DISPLAY FRAME
    cv2.imshow('frame',frame)
    
    # RANDOMLY SAVE FRAME
    if random.random() < 0.025:
        outfile = '{}\\{}_{}.png'.format(folder, frame_no, TimeTag())
        
        # RANDOMLY FLIP FRAME
        if round(random.random()) == 1:
            frame = np.fliplr(frame)
        cv2.imwrite(outfile, frame)
        
    if cv2.waitKey(1) & 0xFF == ord(stop_camera_key):
        break
        
    frame_no += 1

# DISCONNECT CAMERA FEED
cap.release()
cv2.destroyAllWindows()