# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 17:55:11 2019

@author: Not Your Computer
"""

import os
import time
import pandas as pd
import numpy as np
from numpy.random import seed
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from tensorflow import set_random_seed
from skimage.transform import resize
import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
seed(1)
set_random_seed(1)
np.random.seed(1)

# =============================================================================
# FUNCTIONS
# =============================================================================

def regressorCNN(s, n_classes = 2, channels = 3, opt_type = 'adagrad', losses = 'mean_squared_error', final_act = 'sigmoid', learn_rate = 0.001, dropout = True):
    model = Sequential()
    model.add(Conv2D(64, (3,3), input_shape = (s,s,channels), activation = 'relu', data_format = 'channels_last'))
    model.add(Conv2D(64, (3,3), activation = 'relu'))       # CONV 2D
    model.add(Conv2D(64, (3,3), activation = 'relu'))       # CONV 2D
    model.add(MaxPooling2D(pool_size = (2,2)))              # 64 ==> 32
    model.add(Dropout(0.33))                                # DROP
    
    model.add(Conv2D(128, (3,3), activation = 'relu'))      # CONV 2D
    model.add(Conv2D(128, (3,3), activation = 'relu'))      # CONV 2D
    model.add(Conv2D(128, (3,3), activation = 'relu'))      # CONV 2D
    model.add(MaxPooling2D(pool_size = (2,2)))              # 32 ==> 16
    model.add(Dropout(0.33))                                # DROP
    
    model.add(Conv2D(64, (3,3), activation = 'relu'))       # CONV 2D
    model.add(Conv2D(64, (3,3), activation = 'relu'))       # CONV 2D
    model.add(Conv2D(64, (3,3), activation = 'relu'))       # CONV 2D
    model.add(MaxPooling2D(pool_size = (2,2)))              # 16 ==> 8
    model.add(Dropout(0.33))  
    
    model.add(Flatten())                                    # FLATTEN
    
    model.add(Dense(576, activation = 'relu'))              # DENSE
    model.add(Dropout(0.40))                                # DROP
    
    model.add(Dense(64, activation = 'relu'))               # DENSE
    model.add(Dropout(0.40))                                # DROP
    model.add(Dense(1, activation = final_act))             # DENSE
    # Define Optimizer
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
    # Compile Model
    model.compile(loss = losses, optimizer = opt, metrics = ['accuracy'])
    return model


def classifierCNN(img_shape, n_classes = 2, channels = 3, opt_type = 'adagrad', losses = 'mean_squared_error', final_act = 'softmax', learn_rate = 0.001, dropout = True):
    model = Sequential()
    model.add(Conv2D(64, (3,3), input_shape = img_shape, activation = 'relu', data_format = 'channels_last'))
    model.add(Conv2D(64, (3,3), activation = 'relu'))       # CONV 2D
    model.add(Conv2D(64, (3,3), activation = 'relu'))       # CONV 2D
    model.add(MaxPooling2D(pool_size = (2,2)))              # 64 ==> 32
    model.add(Dropout(0.33))                                # DROP
    
    model.add(Conv2D(128, (3,3), activation = 'relu'))      # CONV 2D
    model.add(Conv2D(128, (3,3), activation = 'relu'))      # CONV 2D
    model.add(Conv2D(128, (3,3), activation = 'relu'))      # CONV 2D
    model.add(MaxPooling2D(pool_size = (2,2)))              # 32 ==> 16
    model.add(Dropout(0.33))                                # DROP
    
    model.add(Conv2D(64, (3,3), activation = 'relu'))       # CONV 2D
    model.add(Conv2D(64, (3,3), activation = 'relu'))       # CONV 2D
    model.add(Conv2D(64, (3,3), activation = 'relu'))       # CONV 2D
    model.add(MaxPooling2D(pool_size = (2,2)))              # 16 ==> 8
    model.add(Dropout(0.33))                                # DROP
    
    model.add(Flatten())                                    # FLATTEN
                                 
    model.add(Dense(576, activation = 'relu'))              # DENSE
    model.add(Dropout(0.40))                                # DROP
    
    model.add(Dense(64, activation = 'relu'))               # DENSE
    model.add(Dropout(0.40))                                # DROP
    model.add(Dense(n_classes, activation = final_act))     # DENSE
    # Define Optimizer
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
    # Compile Model
    model.compile(loss = losses, optimizer = opt, metrics = ['categorical_accuracy'])
    return model


def codifyModel(m_type = '', color_type = '', opt_type = '', final_act = '', lr = '', acc = '', auc = ''):
    model_code = ''; parameter_list = []
    model_param = m_type
    if model_param != '':
        if model_param == 'classification':
            model_param = 'CLS'
        elif model_param == 'regression':
            model_param = 'REG'
        parameter_list.append(model_param)
    model_param = color_type
    if model_param != '':
        if model_param == True:
            model_param = 'RGB'
        elif model_param == False:
            model_param = 'GrS'
        parameter_list.append(model_param)
    model_param = opt_type
    if model_param != '':
        parameter_list.append(model_param)
    model_param = final_act
    if model_param != '':
        parameter_list.append(model_param)
    model_param = lr
    if model_param != '':
        model_param = str(model_param)
        model_p_parts = model_param.split('.')
        model_param = model_p_parts[-1]
        parameter_list.append(model_param)
    model_param = acc
    if model_param != '':
        model_param = model_param * 100
        model_param = round(model_param)
        model_param = str(model_param)
        model_p_parts = model_param.split('.')
        model_param = model_p_parts[0]
        parameter_list.append(model_param)
    model_param = auc
    if model_param != '':
        model_param = model_param * 100
        model_param = round(model_param)
        model_param = str(model_param)
        model_p_parts = model_param.split('.')
        model_param = model_p_parts[0]
        parameter_list.append(model_param)
    model_code = '_'.join(parameter_list)
    parameter_list = [m_type, color_type, opt_type, final_act, lr, acc, auc]
    return model_code


def plotROC(y_valid, y_predicted):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, threshold = roc_curve(y_valid, y_predicted)
    auc_score = auc(fpr, tpr)
    # Zoom in view of the upper left corner.
    plt.figure(figsize = (9,6))
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--', lw = 2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim(left = 0, right = 1)
    plt.ylim(top = 1, bottom = 0)
    return auc_score


def plotLayer(model, layer_idx, img, cmap = 'viridis', normalize = False):
    h,w,d = np.shape(img)
    img = img * 255
    m = Model(inputs = model.input, outputs = model.get_layer(index = layer_idx).output)
    p = m.predict(np.reshape(img, (1,h,w,d)))
    img_stack = []
    p_size = np.shape(p)[3]
    for i in range(p_size):
        p_out = p[0,:,:,i]
        img_stack.append(p_out)
    gridPlot(img_stack) 
    return


def gridPlot(img_stack, outfile = 'grid_plot.png'):
    n_imgs = len(img_stack)
    nrows = int(np.floor(np.sqrt(n_imgs)))
    ncols = int(np.ceil(n_imgs / nrows))
    F = plt.figure(figsize = (35,35))
    F.subplots_adjust(left = 0.05, right = 0.95)
    grid = ImageGrid(F, 142, nrows_ncols = (nrows, ncols), axes_pad = 0.0, share_all = True,
                     label_mode = "L", cbar_location = "top", cbar_mode = "single")
    i = 0
    for img in img_stack:
        im = grid[i].imshow(img, interpolation = "nearest", vmin = 0, vmax = 255)
        i += 1 
    grid.cbar_axes[0].colorbar(im)
    plt.savefig(outfile)
    for cax in grid.cbar_axes:
        cax.toggle_label(True)
    return


def getImgPaths(folder_paths, color_imgs = True, s = 64):
    img_paths = []; class_names = []; classes = []
    i = 0
    for folder_path in folder_paths:
        folder_name = os.path.basename(folder_path)
        class_names.append(folder_name)
        for f in os.listdir(folder_path):
            img_path = '{}\\{}'.format(folder_path, f)
            if img_path[-4:].lower() == '.png':
                img_paths.append(img_path)
                classes.append(i)
        i += 1
    return img_paths, classes


def getImgStack(folder_paths, color_imgs = True, s = 64):
    img_stack = []; class_names = []; classes = []
    i = 0
    for folder_path in folder_paths:
        folder_name = os.path.basename(folder_path)
        class_names.append(folder_name)
        for f in os.listdir(folder_path):
            img_path = '{}\\{}'.format(folder_path, f)
            if img_path[-4:].lower() == '.png':
                img = plt.imread(img_path)
                resized_img = resize(img, (s,s))
                img_stack.append(resized_img)
                classes.append(i)
        i += 1
    return img_stack, classes


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def processImage(img_path, anchor_df):
    # CREATE SUBFRAMES AND ASSIGN CLASSES
    sf_classes = []
    bboxes = pd.read_csv("D:\\Images\\BBoxes.csv")
    if img_path in list(bboxes['fpath']):
        bboxes = bboxes[bboxes['fpath'] == img_path]
        img = plt.imread(img_path)
        sf_stack = splitFrame(img, anchor_df)
        i = 0
        for idx,row in anchor_df.iterrows():
            x1,x2,y1,y2 = row[['x1','x2','y1','y2']]
            x1b,y1b,h,w = bboxes[['x','y','h','w']]
            x2b = x1b + w
            y2b = y1b + h
            if IOU([x1,y1,x2,y2], [x1b,y1b,x2b,y2b]) > 0.33:
                sf_classes.append(bboxes['cat'][0])
            else:
                sf_classes.append('nothing')
            i += 1
        return sf_stack, sf_classes
    else:
        return [], []


def Intersection(boxA, boxB):
    # CALCULATE THE OVERLAPPING AREA OF TWO RECTANGLES
    # boxA and boxB are arrays of [x1,y1,x2,y2], where (x2,y2) is the lower right corner
    x1, y1 = (max(boxA[0],boxB[0]), max(boxA[1],boxB[1]))
    x2, y2 = (min(boxA[2],boxB[2]), min(boxA[3],boxB[3]))
    if x2-x1 > 0 and y2-y1 > 0:
        return (x2-x1) * (y2-y1)
    else:
        return 0


def Union(boxA, boxB):
    # CALCULATE THE UNIONED AREA OF TWO RECTANGLES
    # boxA and boxB are arrays of [x1,y1,x2,y2], where (x2,y2) is the lower right corner
    boxA_area = abs((boxA[0]-boxA[2])*(boxA[1]-boxA[3]))
    boxB_area = abs((boxB[0]-boxB[2])*(boxB[1]-boxB[3]))
    if Intersection(boxA, boxB) > 0:
        U = boxA_area + boxB_area - Intersection(boxA, boxB)
        return U
    else:
        return 0


def IOU(boxA, boxB):
    # CALCULATE THE INTERSECTION OVER UNION (IOU) OF TWO RECTANGLES
    # boxA and boxB are arrays of [x1,y1,x2,y2], where (x2,y2) is the lower right corner
    U = Union(boxA, boxB)
    if U <= 0:
        return 0
    else:
        return Intersection(boxA, boxB) / U


def calcAnchors(img_h, img_w, s, overlap = None, outfile = None):
    print('\n[calcAnchors]: Calculating Search Boundaries...')
    anchor_df = []
    
    if overlap == None:
        overlap = 0
        
    nx = int(np.floor((img_w-s*overlap)/(s*(1-overlap))))
    ny = int(np.floor((img_h-s*overlap)/(s*(1-overlap)))) - 1
    sf_no = 0
    
    for scale in search_scales:
        x1 = 0; y1 = 0
        S = round(s * scale)
        nx = int(np.floor((img_w-S*overlap)/(S*(1-overlap))))
        ny = int(np.floor((img_h-S*overlap)/(S*(1-overlap))))

        for i in range(ny):
            y1 = y1 + round(S * (1-overlap))
            y2 = y1 + S

            for j in range(nx):
                x1 = j * round(S * (1-overlap))
                x2 = x1 + S
                
                if x2 > img_w or y2 > img_h:
                    y2 = img_h
                    x2 = img_w
                    x1 = x2 - S
                    y1 = y2 - S
                    
                new_row = [sf_no, S, scale, x1, x2, y1, y2]
                anchor_df.append(new_row) 
                sf_no += 1
                
    anchor_df = pd.DataFrame(anchor_df, columns = ['subframe','size','scale','x1','x2','y1','y2'])
    if outfile != None:
        anchor_df.to_csv(outfile, index = False)
        print('\n[calcBoundaries]: Boundaries written to file.')
    return anchor_df


def splitFrame(frame, anchor_df):
    sf_stack = []
    for i in range(len(anchor_df)):
        s = int(anchor_df['size'][i])
        sf = frame[anchor_df['y1'][i]:anchor_df['y2'][i], anchor_df['x1'][i]:anchor_df['x2'][i], :]
        sf_rows, sf_cols, sf_channels = np.shape(sf)
        if sf_rows > s or sf_cols > s:
            sf = resize(sf, (s,s))
        sf_stack.append(sf)
    return sf_stack


class CNN:
    def __init__(self, model_type = 'classifier', cmode = 'rgb', img_size = 64, size = 'medium',
                 loss_type = 'mean_squared_error', learn_rate = 0.001, final_act = 'sigmoid', opt_type = 'rms', dropout = True, eval_metric = 'accuracy', n_classes = 2):
        self.model_type = model_type; self.learn_rate = learn_rate; self.loss_type = loss_type; self.histories = [];
        self.val_losses = []; self.val_accs = []; self.runtimes = []; self.cmode = cmode; self.img_size = img_size;
        self.trained = False; self.validated = False; self.scores = None; self.epochs = None; self.batch_size = None;
        self.save_name = None;self.final_act = final_act;self.dropout = dropout; self.opt_type = opt_type; self.models = []
        self.metric = eval_metric; self.size = size; self.n_classes = n_classes
        
        if self.cmode == 'rgb':
            self.channels = 3
        else:
            self.channels = 1
            
        return
    
    def train(self, x_train, y_train, x_test, y_test, epochs = 5, batch_size = 32, folds = 1):
        self.epochs = epochs; self.batch_size = batch_size; self.trained = True
        self.acc_list = []; self.loss_list = []; self.folds = folds
        
        x_train = self.loadImages(x_train)
        x_test = self.loadImages(x_test)
        
        if self.model_type == 'classifier':
            if len(np.shape(y_train)) == 1:
                y_train = to_categorical(y_train)
                y_test = to_categorical(y_test)
        print('\n'+'\t'.join([self.model_type, self.final_act, self.loss_type, str(self.epochs), self.opt_type, str(self.learn_rate)]))
        
        if self.folds <= 0:
            print('Number of folds reset to 1.')
            self.folds = 1
            
        for fold in range(self.folds):
            print('\n{} {}'.format(self.model_type.capitalize(), fold+1))
            print('------------------------------------------')

            st = time.time()
            if self.model_type == 'classifier':
                model = classifierCNN(self.img_size, n_classes = self.n_classes, channels = self.channels, opt_type = self.opt_type, losses = self.loss_type, final_act = self.final_act, learn_rate = self.learn_rate, dropout = self.dropout)
            elif self.model_type == 'regressor':
                model = regressorCNN(self.img_size, n_classes = self.n_classes, channels = self.channels, opt_type = self.opt_type, losses = self.loss_type, final_act = self.final_act, learn_rate = self.learn_rate, dropout = self.dropout)
            self.models.append(model)
            self.histories.append(model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, validation_data = (x_test, y_test), shuffle = False))
            if self.model_type == 'regressor':
                self.acc_list.append(self.histories[-1].history['val_acc'])
            else:
                self.acc_list.append(self.histories[-1].history['val_categorical_accuracy'])
            self.loss_list.append(self.histories[-1].history['loss'])
            dt = time.time() - st
            self.runtimes.append(dt)
        self.model = model
        
        if self.folds > 1:
            print('\n{}-Model Summary Stats:'.format(self.folds))
        else:
            print('Training Results:')
        
        print('------------------------------------------')
        print('Accuracy:    {} +/- {}'.format(round(np.mean(self.acc_list),4), round(np.std(self.acc_list),2)))
        print('Loss:        {} +/- {}'.format(round(np.mean(self.loss_list),4), round(np.std(self.loss_list),2)))
        print('Runtime:     {} +/- {}'.format(round(np.mean(self.runtimes),2), round(np.std(self.runtimes),2)))
        self.trained = True
        self.code = codifyModel(m_type = self.model_type, color_type = self.cmode, opt_type = self.opt_type,
                            final_act = self.final_act, lr = self.learn_rate)  
        return
    
    def evaluate(self, x_valid, y_valid, plots_on = True):
        x_valid = self.loadImages(x_valid)
        if self.trained:
            for model in self.models:
                if self.model_type == 'classifier':
                    if len(np.shape(y_valid)) == 1:
                        y_valid = to_categorical(y_valid)
                i = 0
                mean_acc_list = np.zeros(self.epochs)
                mean_loss_list = np.zeros(self.epochs)
                
                for history in self.histories:
                    if self.model_type == 'regressor':
                        acc_list = history.history['val_acc']
                    else:
                        acc_list = history.history['val_categorical_accuracy']
    
                    loss_list = history.history['loss']
                    if i >= 1:
                        for j in range(self.epochs):
                            mean_acc_list[j] += acc_list[j]
                            mean_loss_list[j] += loss_list[j]
                    i+=1
                mean_acc_list = np.divide(mean_acc_list, len(mean_acc_list))
                mean_loss_list = np.divide(mean_loss_list, len(mean_loss_list))
                if plots_on:
                    plt.plot(mean_acc_list)
                    plt.plot(mean_loss_list)
                    plt.show()
                    for history in self.histories:
                        plt.figure(figsize = [7,4])
                        plt.plot(history.history['loss'],'r',linewidth = 2.0, linestyle = '--')
                        plt.plot(history.history['val_loss'],'b',linewidth = 2.0, linestyle = '--')
                        if self.model_type == 'classifier':
                            plt.plot(history.history['categorical_accuracy'],'r',linewidth = 2.0)
                            plt.plot(history.history['val_categorical_accuracy'],'b',linewidth = 2.0)
                        elif self.model_type == 'regressor':
                            plt.plot(history.history['acc'],'r',linewidth = 2.0)
                            plt.plot(history.history['val_acc'],'b',linewidth = 2.0)
                        plt.legend(['Training Data', 'Test Data'], fontsize = 12)
                        plt.xlabel('Epochs', fontsize = 16)
                        plt.ylabel('Loss / Acc',fontsize = 16)
                        plt.title('{} {} {} ({}, {}, {})'.format(self.img_size, self.cmode.upper(), self.model_type.capitalize(), self.opt_type, self.final_act, self.learn_rate), fontsize = 16)
                        plt.show()
                self.scores = model.evaluate(x_valid, y_valid, verbose = 1)
                self.val_losses.append(self.scores[0])
                self.val_accs.append(self.scores[1])
            self.validated = True
            
            # Confusion Matrix
            if self.model_type == 'classifier':
                self.val_acc_list = []; self.prec_list = []; self.spec_list = []
                self.sens_list = []; self.cm_list = []
                print('\nCLASSIFICATION ARCHITECTURE')
                print('------------------------------------------')
                y_valid_reduced = list(np.zeros(len(y_valid)))
                for i in range(len(y_valid)):
                    y_valid_reduced[i] = np.argmax(y_valid[i])
                i = 0
                for model in self.models:
                    model_acc = self.acc_list[i]
#                    cm = confusion_matrix(model.predict_classes(x_valid), y_valid_reduced)
#                    tn, fp, fn, tp = cm.ravel()
#                    model_acc = sum(cm.diagonal()) / cm.sum()
#                    model_prec = tp / (tp + fp)
#                    model_sens = tp / (tp + fn)
                    self.val_acc_list.append(model_acc)
#                    self.prec_list.append(model_prec)
#                    self.sens_list.append(model_sens)
#                    self.cm_list.append(cm)
                    i += 1
                print('Accuracy:           {} +/- {}'.format(round(np.mean(self.val_accs),4), round(np.std(self.val_accs),2)))
                print('Loss:               {} +/- {}'.format(round(np.mean(self.val_losses),4), round(np.std(self.val_losses),2)))

            # ROC & AUC Scores
            if self.model_type == 'regressor':
                self.auc_list = []
                print('\nREGRESSION ARCHITECTURE')
                print('------------------------------------------')
                for model in self.models:
                    fpr, tpr, threshold = roc_curve(y_valid, model.predict(x_valid))
                    auc_score = auc(fpr, tpr)
                    if plots_on:
                        plt.figure(figsize = (9,6))
                        plt.xlim(0, 0.2)
                        plt.ylim(0.8, 1)
                        plt.plot(fpr, tpr)
                        plt.plot([0, 1], [0, 1], 'k--', lw = 2)
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        plt.xlim(left = 0, right = 1)
                        plt.ylim(top = 1, bottom = 0)
                    self.auc_list.append(auc_score)
                print('Accuracy:           {} +/- {}'.format(round(np.mean(self.val_accs),4), round(np.std(self.val_accs),2)))
                print('Loss:               {} +/- {}'.format(round(np.mean(self.val_losses),4), round(np.std(self.val_losses),2)))
                print('AUC:                {} +/- {}'.format(round(np.mean(self.auc_list),4), round(np.std(self.auc_list),2)))
        else:
            print('Model Not Trained!')
        return
    
    def loadImages(self, paths):
        img_stack = []
        for path in paths:
            img = Image.open(path)
            img = np.array(img)
            img_stack.append(img)
        img_stack = np.array(img_stack)
        #h,w,d = np.shape(img)
        #img_stack = np.reshape(img_stack, (h,w,d,1))
        return img_stack
    
    def load(self, fname = 'keras_model.h5'):
        self.models = []
        if 'classif' in fname.lower():
            self.model_type = 'classifier'
        if 'regress' in fname.lower():
            self.model_type = 'regressor'
        self.model = load_model(fname)
        self.models.append(self.model)
        print('Model loaded from: ', fname)
        return
    
    def save(self, fname = 'keras_model.h5'):
        self.model.save(fname)
        self.save_name = fname
        self.fsize = os.path.getsize(fname)
        print('Model saved to: {}\nSize:    {} MB'.format(fname, round(self.fsize/1024/1024,2)))
        return
    
    def plotFilters(self, img, img_class = None):
        if img_class != None:
            print(f'Class: {img_class}')
        plt.imshow(img)
        plt.show()
        print('\nConvolution Outputs')
        i = 0
        for layer in self.model.layers:
            if 'Conv2D' in str(layer):
                plotLayer(self.model, i, img, normalize = False)
            i += 1
        return

def timeTag():
    import datetime
    now = datetime.datetime.now()
    year = str(now.year)[2:]
    month = str(now.month)
    if len(month) == 1:
        month = '0'+month
    day = str(now.day)
    if len(day) == 1:
        day = '0'+day
    hour = str(now.hour)
    if len(hour) == 1:
        hour = '0'+hour
    minute = str(now.minute)
    if len(minute) == 1:
        minute = '0'+minute
    sec = str(now.second)
    if len(sec) == 1:
        sec = '0'+sec
    microsecond = str(now.microsecond)
    time_str = '{}{}{}_{}{}{}_{}'.format(month,day,year,hour,minute,sec,microsecond)
    return time_str

# =============================================================================
# CLASSES
# =============================================================================
    
class CentroidTracker:
    def __init__(self, detect_map):
        return
    
    def register(self, x, y):
        centroid_id = '{}'
        self.active_detect_list.append(centroid_id)
        self.detect_history.append(centroid_id)
        return
    
    def unregister(self, centroid_id):
        self.active_detect_list.pop(centroid_id)
        return