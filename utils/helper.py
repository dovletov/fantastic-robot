import os

from config import *

import numpy as np
import tensorflow as tf
import scipy.io
from random import shuffle, randrange
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from skimage import util
import h5py


# DATASETs
def loadDatasetFromMat(dataset_dir, dataset_name):
    '''
    Load image and corresponding label image from original mat files.
    '''
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
    '''
    Get dataset properties such as height, width, depth of image and
    maximum class number of corresponding label image
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
    
    print('-'*70)
    print("ReLU normalization")
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
    
    print('-'*70)
    print("Tanh normalization")
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
    
    print('-'*70)
    print("Mean normalization")
    print("mean value =\t\t",data.mean())
    print("std value =\t\t",data.std())
    
    return data


# DATA PREPARATION
def getMargin(patch_size):
    '''
    Return patch margin size from a center point.
    '''
    if (patch_size%2 == 0):
        raise ValueError("patch_size should be odd!")
    else:
        margin = (patch_size-1) // 2
        
    return margin
def patchCentered(data, pos_x, pos_y, patch_size):
    '''
    Patch input data of defined size centered at (pos_x, pos_y) 
    coordinates and return it in ChHW (performance optimization)
    '''
    margin = (patch_size-1) // 2
    
    x_top = pos_x - margin
    x_bottom = pos_x + margin+1
    y_left = pos_y - margin
    y_right = pos_y + margin+1

    patch = data[x_top:x_bottom, y_left:y_right, :]
    patch = np.transpose(patch,(2,0,1))
    
    return patch

# tr, vl, ev coords
def generateCoordsList(image, label, patch_size):
    '''
    Form lists of coordinates for each of the classes and stores
    them in another one list.
    '''
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
    '''
    Outputs information about the dataset based on coordinates list.
    '''
    print('-'*70)
    print('\t\t\tlen')
    cl_num = len(coords)
    for cl in range(cl_num):
        cur_coords = coords[cl]
        print("Class "+str(cl+1).zfill(2)+"\t\t"+str(len(cur_coords)))
def splitCoordsListByFrac(coords, vl_frac, ev_frac):
    '''
    '''
    cl_num = len(coords)
    tr_coords, vl_coords, ev_coords = [], [], []

    for cl in range(cl_num):
        
        cur_coords = coords[cl]
        cur_population = len(cur_coords)

        vl_split_size = int(cur_population*vl_frac)
        ev_split_size = int(cur_population*ev_frac)
        tr_split_size = cur_population - vl_split_size - ev_split_size 

        shuffle(cur_coords)

        tr_coords.append(cur_coords[:tr_split_size])
        vl_coords.append(cur_coords[tr_split_size:tr_split_size+vl_split_size])
        ev_coords.append(cur_coords[tr_split_size+vl_split_size:])

    return tr_coords, vl_coords, ev_coords
def splitCoordsByEvCount(image, label, coords, tr_frac, ev_count, patch_size):
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
            index = randrange(*range_index)
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

        shuffle(cur_coords)

        tr_coords.append(cur_coords[:tr_split_size])
        vl_coords.append(cur_coords[tr_split_size:])
            
    return tr_coords, vl_coords, ev_coords
def printSplitInfo(tr_coords, vl_coords, ev_coords):
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
    tr_arr = formArrayFromCoordsList(height, width, tr_coords)
    vl_arr = formArrayFromCoordsList(height, width, vl_coords)
    ev_arr = formArrayFromCoordsList(height, width, ev_coords)

    mul = tr_arr*vl_arr*ev_arr

    if np.sum(mul) > 0:
        raise ValueError('Something wrong with splitting. Intersection detected')

def saveCoordsByFrac(coords_file, tr_coords, vl_coords, ev_coords):
    dictionary = {}
    dictionary["tr_coords"] = tr_coords
    dictionary["vl_coords"] = vl_coords
    dictionary["ev_coords"] = ev_coords
    
    scipy.io.savemat(coords_file, dictionary)
    print('-'*70)
    print('Train, Validation and Evaluation coordinates are saved')
    print(coords_file)
def saveCoordsByEvCount(coords_file, tr_coords, vl_coords, ev_coords):
    dictionary = {}
    dictionary["tr_coords"] = tr_coords
    dictionary["vl_coords"] = vl_coords
    
    ev_obj_array = np.zeros((16,), dtype=np.object)
    for cl in range(16):
        ev_obj_array[cl] = ev_coords[cl]
    dictionary["ev_coords"] = ev_obj_array
    
    scipy.io.savemat(coords_file, dictionary)
    print('-'*70)
    print('Train, Validation and Evaluation coordinates are saved')
    print(coords_file)
def loadCoords(coords_file):
    coords = scipy.io.loadmat(coords_file)
    
    tr_coords = coords['tr_coords'][0]
    vl_coords = coords['vl_coords'][0]
    ev_coords = coords['ev_coords'][0]
    
    print('-'*70)
    print('Train, Validation and Evaluation coordinates are loaded')
    print(coords_file)
    print("lenght tr coords\t",len(tr_coords))
    print("length vl coords\t",len(vl_coords))
    print("length ev coords\t",len(ev_coords))
    return tr_coords, vl_coords, ev_coords
def saveCoordsHdf5(coords_file, tr_coords, vl_coords, ev_coords):
    hf = h5py.File(coords_file, "w")
    hf.create_dataset("tr_coords", data=tr_coords, dtype=np.uint8)
    hf.create_dataset("vl_coords", data=vl_coords, dtype=np.uint8)
    hf.create_dataset("ev_coords", data=ev_coords, dtype=np.uint8)
    hf.close()


# tr, vl, ev patches
def loadPatches(image, patch_size, tr_coords, vl_coords, ev_coords):
    '''
    Loads centered patches based on tr, vl and ev coordinates.
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

def plotArray(array, cl_num, color_list, colorbar=True):
    '''
    Plot array as a ground truth image. If needed include colorbar.
    '''
    plt.figure(figsize=(7, 12))
    mat = plt.imshow(array,
                    cmap=mcolors.ListedColormap(color_list),
                    vmin = 0-.5, 
                    vmax = len(color_list)-1+.5, 
                    alpha=1)
    if colorbar:
        cax = plt.colorbar(mat, ticks=[i for i in range(cl_num+1)])
    plt.show()
