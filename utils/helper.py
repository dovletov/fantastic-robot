import os

from config import *

import numpy as np
import tensorflow as tf

import scipy.io


# DATASETs
def loadDatasetFromMat(dataset_name):
    '''
    Load image and corresponding label image from original mat files
    '''
    if (dataset_name == "ss"):
        image = scipy.io.loadmat(os.path.join(ORIG_DATA_DIR, 'Salinas.mat'))['salinas']
        label = scipy.io.loadmat(os.path.join(ORIG_DATA_DIR, 'Salinas_gt.mat'))['salinas_gt']
        print("---------------- dataset Salinal loaded ----------------")
    elif (dataset_name == "ssc"):
        image = scipy.io.loadmat(os.path.join(ORIG_DATA_DIR, 'Salinas_corrected.mat'))['salinas_corrected']
        label = scipy.io.loadmat(os.path.join(ORIG_DATA_DIR, 'Salinas_gt.mat'))['salinas_gt']
        print("---------------- dataset Salinal CORRECTED loaded ----------------")
    elif (dataset_name == "ssc815"):
        image = scipy.io.loadmat(os.path.join(ORIG_DATA_DIR, 'Salinas_corrected.mat'))['salinas_corrected']
        label = scipy.io.loadmat(os.path.join(ORIG_DATA_DIR, 'Salinas_gt_815.mat'))['salinas_gt_815']
        print("---------------- dataset Salinal CORRECTED loaded ----------------")
    elif (dataset_name == "ssc_pca"):
        image = scipy.io.loadmat(os.path.join(ORIG_DATA_DIR, 'Salinas_corrected_pca.mat'))['salinas_corrected_pca']
        label = scipy.io.loadmat(os.path.join(ORIG_DATA_DIR, 'Salinas_gt.mat'))['salinas_gt']
        print("---------------- dataset Salinal CORRECTED PCA3 loaded ----------------")
    elif (dataset_name == "ip"):
        image = scipy.io.loadmat(os.path.join(ORIG_DATA_DIR, 'Indian_pines.mat'))['indian_pines']
        label = scipy.io.loadmat(os.path.join(ORIG_DATA_DIR, 'Indian_pines_gt.mat'))['indian_pines_gt']
        print("---------------- dataset Indian Pines loaded ----------------")
    elif (dataset_name == "ipc"):
        image = scipy.io.loadmat(os.path.join(ORIG_DATA_DIR, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        label = scipy.io.loadmat(os.path.join(ORIG_DATA_DIR, 'Indian_pines_gt.mat'))['indian_pines_gt']
        print("---------------- dataset Indian Pines CORRECTED loaded ----------------")
    elif (dataset_name == "ipc_pca"):
        image = scipy.io.loadmat(os.path.join(ORIG_DATA_DIR, 'Indian_pines_corrected_pca.mat'))['indian_pines_corrected_pca']
        label = scipy.io.loadmat(os.path.join(ORIG_DATA_DIR, 'Indian_pines_gt.mat'))['indian_pines_gt']
        print("---------------- dataset Indian Pines CORRECTED PCA3 loaded ----------------")
    elif (dataset_name == "pu"):
        image = scipy.io.loadmat(os.path.join(ORIG_DATA_DIR, 'PaviaU.mat'))['paviaU']
        label = scipy.io.loadmat(os.path.join(ORIG_DATA_DIR, 'PaviaU_gt.mat'))['paviaU_gt']
        print("---------------- dataset Pavia University is loaded ----------------")
    elif (dataset_name == "pc"):
        image = scipy.io.loadmat(os.path.join(ORIG_DATA_DIR, 'Pavia.mat'))['pavia']
        label = scipy.io.loadmat(os.path.join(ORIG_DATA_DIR, 'Pavia_gt.mat'))['pavia_gt']
        print("---------------- dataset Pavia Centre is loaded ----------------")
    elif (dataset_name == "ksc"):
        image = scipy.io.loadmat(os.path.join(ORIG_DATA_DIR, 'KSC.mat'))['KSC']
        label = scipy.io.loadmat(os.path.join(ORIG_DATA_DIR, 'KSC_gt.mat'))['KSC_gt']
        print("---------------- dataset Kennedy Space Center is loaded ----------------")
    elif (dataset_name == "b"):
        image = scipy.io.loadmat(os.path.join(ORIG_DATA_DIR, 'Botswana.mat'))['Botswana']
        label = scipy.io.loadmat(os.path.join(ORIG_DATA_DIR, 'Botswana_gt.mat'))['Botswana_gt']
        print("---------------- dataset Botswana is loaded ----------------")
    
    return image, label
def getDatasetProperty(image, label):
    '''
    Get dataset properties such as height, width, depth of image and
    class number of corresponding label image
    '''
    if image.shape[0]==label.shape[0] and image.shape[1]==label.shape[1]:
        height = image.shape[0]
        width = image.shape[1]
        depth = image.shape[2]
        class_number = label.max()
    
    return height, width, depth, class_number


# NORMALIZATIONs
def reluNormalization(data):
    '''
    Normalize data between 0 and 1
    '''
    max_minus_min = data.max() - data.min()
    data = data.astype(float)
    data -= data.min()
    data /= max_minus_min
    
    print("---------------- dataset normalized for ReLU ----------------")
    print("min value =\t\t",data.min())
    print("max value =\t\t",data.max())
    
    return data
def tanhNormalization(data):
    '''
    Normalize data in range between -1 and 1
    '''
    max_minus_min = data.max() - data.min()
    data = data.astype(float)
    data -= data.min()
    data /= max_minus_min
    data *= 2
    data -= 1
    
    print("---------------- dataset normalized for tanh ----------------")
    print("min value =\t\t",data.min())
    print("max value =\t\t",data.max())
    
    return data
def meanNormalization(data):
    '''
    Mean Normalize (Standardize data)
    '''
    std = data.std()
    data = data.astype(float)
    data -= data.mean()
    data /= std
    
    print("---------------- dataset mean normalized ----------------")
    print("mean value =\t\t",data.mean())
    print("std value =\t\t",data.std())
    
    return data


# DATA PREPARATION
def patchCentered(data, pos_x, pos_y, patch_size):
    '''Patch input data of defined size centered at (pos_x, pos_y) 
    coordinates and return it in ChHW (performance optimization)
    '''
    margin = (patch_size-1) // 2
    
    x_left = pos_x-margin
    x_right = pos_x+margin+1
    y_top = pos_y-margin
    y_bottom = pos_y+margin+1

    patch = data[x_left:x_right, y_top:y_bottom, :]
    patch = np.transpose(patch,(2,0,1))
    
    return patch


# TRAIN,VALID,EVAL coords
def saveCoords(coords_file, tr_coords, vl_coords, ev_coords):
    dictionary = {}
    
    dictionary["tr_coords"] = tr_coords
    dictionary["vl_coords"] = vl_coords
    dictionary["ev_coords"] = ev_coords
    
    scipy.io.savemat(coords_file, dictionary)
    print(coords_file)
    print("---------------- TRAIN/VALID/TEST COORDS are saved   ----------------")
def loadCoords(coords_file):
    coords = scipy.io.loadmat(coords_file)
    
    tr_coords = coords['tr_coords'][0]
    vl_coords = coords['vl_coords'][0]
    ev_coords = coords['ev_coords'][0]    
    
    print("---------------- TRAIN/VALID/TEST COORDS are loaded ----------------")
    print("lenght tr coords\t",len(tr_coords))
    print("length vl coords\t",len(vl_coords))
    print("length ev coords\t",len(ev_coords))
    return tr_coords, vl_coords, ev_coords
