from utils.helper import *
from config import *
import argparse
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


# parser
parser = argparse.ArgumentParser(
    description='This is test program.',
    epilog='This is the end of help',
    prog='main')

parser.add_argument('-g', '--gpu_id', 
    default='0', 
    help='''Id number of graphical card used to run %(prog)s. By
            default uses '0''')

parser.add_argument('-i', '--dataset_name', 
    default='ssc', 
    help='''Input dataset. Possible values are: ssc, ss, ip, ipc,
            pc, pu, ksc, b. By default uses 'ssc''')

parser.add_argument('-p', '--n_pca_components',
    type = int,
    default=None,
    help='''Number of pca components to retain. By default 'None'
            value is set, which means that no PCA is used.''')
parser.add_argument('-s', '--patch_size',
    type = int,
    default=5,
    help='''''')
parser.add_argument('-e', '--ev_points_per_class',
    type = int,
    default=int(15))
parser.add_argument('-m', '--mode')
parser.add_argument('-n', '--configuration_name')
parser.add_argument('-a', '--augmentation_on', default=False)
parser.add_argument('-d', '--dropout_on', default=False)
parser.add_argument('-z', '--ckpt_id')


# parsing
args = parser.parse_args()
print('Parsing arguments...')
print("gpu_id\t\t\t", args.gpu_id)
print("dataset_name\t\t", args.dataset_name)
print("n_pca_components\t", args.n_pca_components)
print("patch_size\t\t", args.patch_size)
margin = getMargin(args.patch_size)
print("ev_points_per_class\t", args.ev_points_per_class)
print("mode\t\t\t", args.mode)
print("configuration_name\t", args.configuration_name)
print("augmentation_on\t\t", args.augmentation_on)
print("dropout_on\t\t", args.dropout_on)
print('Parsing arguments finished...')


# Environment Variable
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

# create main folder
configuration_path = os.path.join(RESULTS_DIR, args.configuration_name)
Path(configuration_path).mkdir(parents=True, exist_ok=True)

# input data preprocessing
image, label = loadDatasetFromMat(ORIG_DATA_DIR, args.dataset_name)
height, width, depth, class_number = getDatasetProperty(image, label)
image = meanNormalization(image)

coords = generateCoordsList(image, label, args.patch_size)
printCoordsListInfo(coords)

# load coords file or create if doesn't exist
coords_filename = args.dataset_name + '_s' + \
    str(args.patch_size).zfill(2) + '_e' + \
    str(args.ev_points_per_class).zfill(2) +\
    '.mat'
coords_file = os.path.join(configuration_path, coords_filename)
if Path(coords_file).is_file():
    tr_coords, vl_coords, ev_coords = loadCoords(coords_file)
else:
    tr_coords, vl_coords, ev_coords = splitCoordsListByFrac(coords, 0.2, 0.2)
    saveCoords(coords_file, tr_coords, vl_coords, ev_coords)
printSplitInfo(tr_coords, vl_coords, ev_coords)

tr_patches, vl_patches, ev_patches = loadPatches(image, args.patch_size, tr_coords, vl_coords, ev_coords)
printSplitInfo(tr_patches, vl_patches, ev_patches)

# plt.figure(figsize=(7, 12))
mat = plt.imshow(label,
                cmap=mcolors.ListedColormap(COLORS_D17),
                vmin = 0-.5, 
                vmax = len(COLORS_D17)-1+.5, 
                alpha=1)
# cax = plt.colorbar(mat, ticks=np.unique(label))
plt.show()