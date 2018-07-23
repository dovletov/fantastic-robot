import os

from config import *

import numpy as np
import tensorflow as tf

import scipy.io



def loadDatasetFromMat(dataset_name):
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
    if image.shape[0]==label.shape[0] and image.shape[1]==label.shape[1]:
        height  = image.shape[0]
        width   = image.shape[1]
        depth   = image.shape[2]
        class_number = label.max()
    return height, width, depth, class_number
