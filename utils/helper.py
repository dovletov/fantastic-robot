import os

from config import *

import numpy as np
import tensorflow as tf
import scipy.io
import random
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from skimage import util
import tensorflow.contrib.slim as slim
from tensorflow import initializers as tfinit


# DATASETs
def loadDatasetFromMat(dataset_dir, dataset_name):
    """
    Load image and corresponding label image from original mat files.
    """
    print('-'*70)
    if (dataset_name == "ss"):
        image = scipy.io.loadmat(os.path.join(dataset_dir, \
            'Salinas.mat'))['salinas']
        label = scipy.io.loadmat(os.path.join(dataset_dir, \
            'Salinas_gt.mat'))['salinas_gt']
        print("Salinal dataset is loaded")
    elif (dataset_name == "ssc"):
        image = scipy.io.loadmat(os.path.join(dataset_dir, \
            'Salinas_corrected.mat'))['salinas_corrected']
        label = scipy.io.loadmat(os.path.join(dataset_dir, \
            'Salinas_gt.mat'))['salinas_gt']
        print("Salinal CORRECTED dataset is loaded")
    elif (dataset_name == "ssc815"):
        image = scipy.io.loadmat(os.path.join(dataset_dir, \
            'Salinas_corrected.mat'))['salinas_corrected']
        label = scipy.io.loadmat(os.path.join(dataset_dir, \
            'Salinas_gt_815.mat'))['salinas_gt_815']
        print("Salinal CORRECTED 815 dataset is loaded")
    elif (dataset_name == "ssc_pca"):
        image = scipy.io.loadmat(os.path.join(dataset_dir, \
            'Salinas_corrected_pca.mat'))['salinas_corrected_pca']
        label = scipy.io.loadmat(os.path.join(dataset_dir, \
            'Salinas_gt.mat'))['salinas_gt']
        print("Salinal CORRECTED PCA3 dataset is loaded")
    elif (dataset_name == "ip"):
        image = scipy.io.loadmat(os.path.join(dataset_dir, \
            'Indian_pines.mat'))['indian_pines']
        label = scipy.io.loadmat(os.path.join(dataset_dir, \
            'Indian_pines_gt.mat'))['indian_pines_gt']
        print("Indian Pines dataset is loaded")
    elif (dataset_name == "ipc"):
        image = scipy.io.loadmat(os.path.join(dataset_dir, \
            'Indian_pines_corrected.mat'))['indian_pines_corrected']
        label = scipy.io.loadmat(os.path.join(dataset_dir, \
            'Indian_pines_gt.mat'))['indian_pines_gt']
        print("Indian Pines CORRECTED dataset is loaded")
    elif (dataset_name == "ipc_pca"):
        image = scipy.io.loadmat(os.path.join(dataset_dir, \
            'Indian_pines_corrected_pca.mat'))['indian_pines_corrected_pca']
        label = scipy.io.loadmat(os.path.join(dataset_dir, \
            'Indian_pines_gt.mat'))['indian_pines_gt']
        print("Indian Pines CORRECTED PCA3 dataset is loaded")
    elif (dataset_name == "pu"):
        image = scipy.io.loadmat(os.path.join(dataset_dir, \
            'PaviaU.mat'))['paviaU']
        label = scipy.io.loadmat(os.path.join(dataset_dir, \
            'PaviaU_gt.mat'))['paviaU_gt']
        print("Pavia University dataset is loaded")
    elif (dataset_name == "pc"):
        image = scipy.io.loadmat(os.path.join(dataset_dir, \
            'Pavia.mat'))['pavia']
        label = scipy.io.loadmat(os.path.join(dataset_dir, \
            'Pavia_gt.mat'))['pavia_gt']
        print("Pavia Centre dataset is loaded")
    elif (dataset_name == "ksc"):
        image = scipy.io.loadmat(os.path.join(dataset_dir, \
            'KSC.mat'))['KSC']
        label = scipy.io.loadmat(os.path.join(dataset_dir, \
            'KSC_gt.mat'))['KSC_gt']
        print("Kennedy Space Center dataset is loaded")
    elif (dataset_name == "b"):
        image = scipy.io.loadmat(os.path.join(dataset_dir, \
            'Botswana.mat'))['Botswana']
        label = scipy.io.loadmat(os.path.join(dataset_dir, \
            'Botswana_gt.mat'))['Botswana_gt']
        print("Botswana dataset is loaded")
    
    return image, label
def getDatasetProperty(image, label):
    """
    Get dataset properties such as height, width, depth of image and
    maximum class number of corresponding label image
    """
    if image.shape[0]==label.shape[0] and image.shape[1]==label.shape[1]:
        height = image.shape[0]
        width = image.shape[1]
        depth = image.shape[2]
        class_number = label.max()
    
    return height, width, depth, class_number


# NORMALIZATIONs
def reluNormalization(data):
    """
    Normalize data between 0 and 1
    """
    max_minus_min = data.max() - data.min()
    data = data.astype(float)
    data -= data.min()
    data /= max_minus_min
    
    print('-'*70)
    print("ReLU normalization")
    print("min value =\t\t",data.min())
    print("max value =\t\t",data.max())
    
    return data
def tanhNormalization(data):
    """
    Normalize data in range between -1 and 1
    """
    max_minus_min = data.max() - data.min()
    data = data.astype(float)
    data -= data.min()
    data /= max_minus_min
    data *= 2
    data -= 1
    
    print('-'*70)
    print("Tanh normalization")
    print("min value =\t\t",data.min())
    print("max value =\t\t",data.max())
    
    return data
def meanNormalization(data):
    """
    Mean Normalize (Standardize data)
    """
    std = data.std()
    data = data.astype(float)
    data -= data.mean()
    data /= std
    
    print('-'*70)
    print("Mean normalization")
    print("mean value =\t\t",data.mean())
    print("std value =\t\t",data.std())
    
    return data


# DATA PREPARATION
def getMargin(patch_size):
    """
    Return patch margin size from a center point.
    """
    if (patch_size%2 == 0):
        raise ValueError("patch_size should be odd!")
    else:
        margin = (patch_size-1) // 2
        
    return margin
def patchCentered(data, pos_x, pos_y, patch_size):
    """
    Patch input data of defined size centered at (pos_x, pos_y) 
    coordinates and return it in HWC.
    """
    margin = (patch_size-1) // 2
    
    x_top = pos_x - margin
    x_bottom = pos_x + margin+1
    y_left = pos_y - margin
    y_right = pos_y + margin+1

    patch = data[x_top:x_bottom, y_left:y_right, :]
    
    return patch

# tr, vl, ev coords
def generateCoordsList(image, label, patch_size):
    """
    Form lists of coordinates for each of the classes and stores
    them in another one list.
    """
    h, w, d, cl_num = getDatasetProperty(image, label)
    m = getMargin(patch_size)

    coords = []
    for cl in range(cl_num):
        coords.append([])
    for x in range(m, h - m):
        for y in range(m, w - m):
            curr_tar = label[x,y]
            if curr_tar > 0:
                coords[curr_tar-1].append([x,y])
    return coords
def printCoordsListInfo(coords):
    """
    Outputs information about dataset based on the list of coordinates.
    """
    print('-'*70)
    print('\t\t\tlen')
    cl_num = len(coords)
    for cl in range(cl_num):
        cur_coords = coords[cl]
        print("Class "+str(cl+1).zfill(2)+"\t\t"+str(len(cur_coords)))
def splitCoordsListByFrac(coords, vl_frac, ev_frac):
    """
    Splits labeled part of image into train, validation and evaluation subsets
    based on 'vl_frac' and 'ev_frac'
    """
    cl_num = len(coords)
    tr_coords, vl_coords, ev_coords = [], [], []

    for cl in range(cl_num):
        
        cur_coords = coords[cl]
        cur_population = len(cur_coords)

        vl_split_size = int(cur_population*vl_frac)
        ev_split_size = int(cur_population*ev_frac)
        tr_split_size = cur_population - vl_split_size - ev_split_size 

        random.shuffle(cur_coords)

        tr_coords.append(cur_coords[:tr_split_size])
        vl_coords.append(cur_coords[tr_split_size:tr_split_size+vl_split_size])
        ev_coords.append(cur_coords[tr_split_size+vl_split_size:])

    return tr_coords, vl_coords, ev_coords
def splitCoordsByEvCount(image, label, coords, tr_frac, ev_count, patch_size):
    """
    Splits labeled part of image into train, validation and evaluation subsets.
    First forms evaluation subset with 'ev_count' coordinates for each class 
    and guarantees that there would be no overlapping within evaluation patches.
    After that forms training and validation subsets of coordinates based on
    'tr_frac'. Based on them training and validation patches can intersect with 
    each other.
    """
    h, w, d, cl_num = getDatasetProperty(image, label)
    m = getMargin(patch_size)

    # gt image
    # mat = plt.imshow(label, 
    #                  cmap=mcolors.ListedColormap(COLORS_D17),
    #                  vmin = 0-.5, 
    #                  vmax = len(COLORS_D17)-1+.5, 
    #                  alpha=1)
    # cax = plt.colorbar(mat, ticks=np.unique(label))
    # plt.show()

    ev_map = np.zeros((h, w), dtype=np.uint8)
    ev_map_with_margin = np.zeros((h, w), dtype=np.uint8)
    temp_map = np.zeros((h, w), dtype=np.uint8)
    
    # extract count evaluation points
    ev_coords = []
    for cl in range(cl_num):
        ev_coords.append([])
        counter = 0
        cur_cl_coors = coords[cl]
        range_index = (0, len(cur_cl_coors))
        while counter < ev_count:
            temp_map = temp_map*0
            index = random.randrange(*range_index)
            x, y = cur_cl_coors[index]
            temp_map[x-m:x+m+1, y-m:y+m+1] = 1
            if np.sum(ev_map*temp_map) == 0:
                ev_coords[cl].append([x, y])
                ev_map[x-m:x+m+1, y-m:y+m+1] = 1
                ev_map_with_margin[x-2*m:x+2*m+1, y-2*m:y+2*m+1] = 1
                counter += 1
    
    # plt.imshow(ev_map, 
    #            cmap=mcolors.ListedColormap(["white", "black"]), 
    #            alpha=0.5)
    # plt.show()

    ev_map_with_margin_inv = util.invert(ev_map_with_margin.astype(float))
    tr_vl_label = label*ev_map_with_margin_inv.astype(int)

    tr_vl_coords = []
    for i in range(cl_num):
        tr_vl_coords.append([])

    for x in range(m, h - m):
        for y in range(m, w - m):
            curr_tar = tr_vl_label[x,y]
            if curr_tar > 0:
                tr_vl_coords[curr_tar-1].append([x,y])


    tr_coords, vl_coords = [], []
    for cl in range(cl_num):
        cur_coords = tr_vl_coords[cl]
        cur_population = len(tr_vl_coords[cl])

        tr_split_size = int(cur_population*tr_frac)

        random.shuffle(cur_coords)

        tr_coords.append(cur_coords[:tr_split_size])
        vl_coords.append(cur_coords[tr_split_size:])
            
    return tr_coords, vl_coords, ev_coords
def printSplitInfo(tr_coords, vl_coords, ev_coords):
    """
    Prints split information for each class.
    """
    cl_num = len(tr_coords)
    print('-'*70)
    print('\t\t\tlen(tr)\t\tlen(vl)\t\tlen(ev)\t\tlen(sum)')
    for cl in range(cl_num):
        print("Class%s \t\t %s \t\t %s \t\t %s \t\t"% (\
            str(cl+1).zfill(2),
            str(len(tr_coords[cl])).zfill(5),
            str(len(vl_coords[cl])).zfill(5),
            str(len(ev_coords[cl])).zfill(5)))
        # print(str(len(tr_coords[cl]+len(vl_coords[cl])))); quit()
        
def formArrayFromCoordsList(height, width, coords):
    """
    Generates ground truth array based on coordinate list.
    """
    array = np.zeros((height, width), dtype=np.uint8)
    
    cl_num = len(coords)
    for cl in range(cl_num):
        current_coords = coords[cl]
        count = len(coords[cl])
        for i in range(count):
            x = current_coords[i][0]
            y = current_coords[i][1]
            array[x,y] = cl + 1
    return array
def splitChecker(height, width, tr_coords, vl_coords, ev_coords):
    """
    Checks whether there is some intersection between train, validation and
    evaluation coordinate lists. In case if there is one raises ValueError.
    """
    tr_arr = formArrayFromCoordsList(height, width, tr_coords)
    vl_arr = formArrayFromCoordsList(height, width, vl_coords)
    ev_arr = formArrayFromCoordsList(height, width, ev_coords)

    mul = tr_arr*vl_arr*ev_arr

    if np.sum(mul) > 0:
        raise ValueError('Something wrong with splitting. Intersection detected')

def saveCoords(coords_file, tr_coords, vl_coords, ev_coords):
    """
    Saves training, validation and evaluation coordinate lists into
    mat file. Dataset are saved as class 'cell'.  
    """
    dictionary = {}
    
    len_tr_coords = len(tr_coords)
    tr_obj_array = np.zeros((len_tr_coords,), dtype=np.object)
    for cl in range(len_tr_coords):
        tr_obj_array[cl] = tr_coords[cl]
    dictionary["tr_coords"] = tr_obj_array

    len_vl_coords = len(vl_coords)
    vl_obj_array = np.zeros((len_vl_coords,), dtype=np.object)
    for cl in range(len_vl_coords):
        vl_obj_array[cl] = vl_coords[cl]
    dictionary["vl_coords"] = vl_obj_array

    len_ev_coords = len(ev_coords)
    ev_obj_array = np.zeros((len_ev_coords,), dtype=np.object)
    for cl in range(len_ev_coords):
        ev_obj_array[cl] = ev_coords[cl]
    dictionary["ev_coords"] = ev_obj_array
    
    scipy.io.savemat(coords_file, dictionary)
    print('-'*70)
    print('Train, Validation and Evaluation coordinates are saved')
    print(coords_file)
def loadCoords(coords_file):
    """
    Loads train, validation and evaluation coordinate lists from mat
    file. Index '0' is added in order to load the content of the cells.
    """
    coords = scipy.io.loadmat(coords_file)
    
    tr_coords = coords['tr_coords'][0]
    vl_coords = coords['vl_coords'][0]
    ev_coords = coords['ev_coords'][0]
    
    print('-'*70)
    print('Train, Validation and Evaluation coordinates are loaded')
    print(coords_file)

    return tr_coords, vl_coords, ev_coords


# tr, vl, ev patches
def loadPatches(image, patch_size, tr_coords, vl_coords, ev_coords):
    '''
    Loads centered patches based on train, validation and evaluation 
    coordinate lists.
    '''
    tr_patches, vl_patches, ev_patches = [], [], []

    for cl in range(len(tr_coords)):
        tr_patches.append([])
        cur_cl_coords = tr_coords[cl]
        cur_cl_patches = tr_patches[cl]
        for i in range(len(cur_cl_coords)):
            x, y = cur_cl_coords[i]
            cur_cl_patches.append(patchCentered(image, x, y, patch_size))

    for cl in range(len(vl_coords)):
        vl_patches.append([])
        cur_cl_coords = vl_coords[cl]
        cur_cl_patches = vl_patches[cl]
        for i in range(len(cur_cl_coords)):
            x, y = cur_cl_coords[i]
            cur_cl_patches.append(patchCentered(image, x, y, patch_size))

    for cl in range(len(ev_coords)):
        ev_patches.append([])
        cur_cl_coords = ev_coords[cl]
        cur_cl_patches = ev_patches[cl]
        for i in range(len(cur_cl_coords)):
            x, y = cur_cl_coords[i]
            cur_cl_patches.append(patchCentered(image, x, y, patch_size))

    return tr_patches, vl_patches, ev_patches
def simpleAugmentation(patch_list):
    """
    Performs simple augmentation (90, 180 and 270 degrees) and shuffles
    new augmented lists. 
    Uses np.rot90 since it is faster that skimage.transform.rotate.
    """
    cl_num = len(patch_list)
    
    augmented = []
    for cl in range(cl_num):
        augmented.append([])
        cur_cl_patches = patch_list[cl]
        for patch in cur_cl_patches:
            augmented[cl].append(patch)
            augmented[cl].append(np.rot90(patch, k=1, axes=(1,2)))
            augmented[cl].append(np.rot90(patch, k=2, axes=(1,2)))
            augmented[cl].append(np.rot90(patch, k=3, axes=(1,2)))
        random.shuffle(augmented[cl])

    return augmented
def plotArray(array, cl_num, color_list, colorbar=True):
    """
    Plot array as a ground truth image. If needed include colorbar.
    """
    plt.figure(figsize=(7, 12))
    mat = plt.imshow(array,
                    cmap=mcolors.ListedColormap(color_list),
                    vmin = 0-.5, 
                    vmax = len(color_list)-1+.5, 
                    alpha=1)
    if colorbar:
        cax = plt.colorbar(mat, ticks=[i for i in range(cl_num+1)])
    plt.show()

# patch, batch generation
def getRandPatch(patch_list, order):
    """
    Returns randomly choosen patch and corresponding ground truth value.
    'NHWC' order for 2DCNN, where C is the depth of input patch.
    'NDHWC' order for 3DCNN, where D is the depth of input patch.
    """
    cl_num = len(patch_list)

    cl = random.randint(0, cl_num-1)
    idx = random.randint(0, len(patch_list[cl])-1)
    
    if order == 'NHWC':
        patch = patch_list[cl][idx]
    elif order == 'NDHWC':
        # HWC -> DHW -> DHW1
        patch = patch_list[cl][idx]
        patch = np.transpose(patch,(2,0,1))
        patch = np.expand_dims(patch, axis=3)

    target = cl

    patch = np.expand_dims(patch, 0)
    target = np.expand_dims(target, 0)

    return patch, target
def getRandBatch(patch_list, batch_size, order):
    """
    Randomly forms batch and corresponding ground truth list.
    """
    image_batch, target_batch = getRandPatch(patch_list, order=order)
    for i in range(batch_size-1):
        image, target = getRandPatch(patch_list, order = order)
        image_batch = np.concatenate((image_batch, image), axis=0)
        target_batch = np.concatenate((target_batch, target), axis=0)

    return image_batch, target_batch

# evaluation
def getPredictionAcc(batch_prediction, targets):
    """
    Calculate accuracy value for the batch of predictions.
    """
    batch_size = batch_prediction.shape[0]

    logits = np.argmax(batch_prediction, axis=1)
    num_correct = np.sum(np.equal(logits, targets))
    
    acc = 100. * (num_correct/batch_size)

    return acc


# placeholders
def declarePlaceholder2D(patch_size, depth, order='NHWC'):
    """
    Declare image and target placeholders for 2D CNN. Input image placeholder
    is in NHWC order. Target image placeholder is of size B, since only one
    label can be assigned in classification taks.
    """
    if order == 'NHWC':
        x_input_shape = (None, patch_size, patch_size, depth)
    x_input = tf.placeholder(tf.float32, 
        shape = x_input_shape,
        name = 'input_image')
    Y_target = tf.placeholder(tf.int32, 
        shape = [None],
        name = 'target_label')
    return x_input, Y_target
def declarePlaceholder3D(patch_size, depth, order='NDHWC'):
    """
    Declare image and target placeholders for 3D CNN. Input image placeholder
    is in NDHWC order. Target image placeholder is of size N, since only one
    label can be assigned in classification taks.
    """
    if order == 'NDHWC':
        x_input_shape = (None, depth, patch_size, patch_size, 1)
    x_input = tf.placeholder(tf.float32, 
        shape=x_input_shape,
        name = 'input_image')
    Y_target = tf.placeholder(tf.int32, 
        shape = [None],
        name = 'target_label')
    return x_input, Y_target

