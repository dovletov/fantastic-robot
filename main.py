from utils.helper import *
from utils.models import cnn2d_example, cnn3d_example
from config import *
import argparse
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import tensorflow as tf
from tensorflow.python.framework import ops


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
parser.add_argument('-a', '--augmentation_on', 
    default=False)
parser.add_argument('-d', '--dropout_on', 
    default=False)
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
plotArray(label, class_number, COLORS_D17, colorbar=True)

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
    # tr_coords, vl_coords, ev_coords = splitCoordsListByFrac(coords, 0.2, 0.2)
    tr_coords, vl_coords, ev_coords = splitCoordsByEvCount(image, label, coords, 0.7, args.ev_points_per_class, args.patch_size)
    saveCoords(coords_file, tr_coords, vl_coords, ev_coords)
printSplitInfo(tr_coords, vl_coords, ev_coords)
splitChecker(height, width, tr_coords, vl_coords, ev_coords)

# load patches and perform augmentation if needed.
tr_patches, vl_patches, ev_patches = loadPatches(image, args.patch_size, tr_coords, vl_coords, ev_coords)
if args.augmentation_on:
    tr_patches = simpleAugmentation(tr_patches)
    printSplitInfo(tr_patches, vl_patches, ev_patches)

#  drouput values
if args.dropout_on:
    pkeep_conv_val = 0.8
    pkeep_hidden_val = 0.8
else:
    pkeep_conv_val = 1
    pkeep_hidden_val = 1



ops.reset_default_graph()

# x, Y = declarePlaceholder2D(args.patch_size, depth, order='NHWC')
x, Y = declarePlaceholder3D(args.patch_size, depth, order='NDHWC')

global_step = tf.Variable(0, name='global_step', trainable=False)
pkeep_conv = tf.placeholder("float", name = 'pkeep_conv')
pkeep_hidden = tf.placeholder("float", name = 'pkeep_hidden')

with tf.variable_scope('model') as model:
    # output = cnn2d_example(x, pkeep_conv, pkeep_hidden)
    output = cnn3d_example(x, pkeep_conv, pkeep_hidden)

with tf.variable_scope('loss') as loss:
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=Y))

with tf.variable_scope('optimizer') as optimizer:
    optimizer = tf.train.MomentumOptimizer(0.005, 0.9)
    train_step = optimizer.minimize(loss, global_step=global_step)

with tf.variable_scope('prediction') as prediction:
    prediction = tf.nn.softmax(output)


print(tr_patches[0][0].shape)
# Allowing GPU memory growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

generations = 8000
batch_size = 100

with tf.Session(config=config) as sess:

    if args.mode == 'train':
        
        # initialize variables
        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(generations):
            # order = 'NHWC' # 2D CNN
            order = 'NDHWC' # 3D CNN
            tr_x, tr_Y = getRandBatch(tr_patches, batch_size, order=order)
            tr_dict = {x: tr_x, Y: tr_Y, pkeep_conv: pkeep_conv_val, pkeep_hidden: pkeep_hidden_val }

            sess.run(train_step, feed_dict=tr_dict)
            # global_step_val = sess.run(global_step)
            
            if (i+1) % 50 == 0:
                tmp_tr_loss, tmp_tr_pr = sess.run([loss, prediction], feed_dict=tr_dict)
                tmp_tr_acc = getPredictionAcc(tmp_tr_pr, tr_Y)

                vl_x, vl_Y = getRandBatch(vl_patches, batch_size, order=order)
                vl_dict = {x: vl_x, Y: vl_Y, pkeep_conv: 1.0, pkeep_hidden: 1.0}
                tmp_vl_loss, tmp_vl_pr = sess.run([loss, prediction], feed_dict=vl_dict)
                tmp_vl_acc = getPredictionAcc(tmp_vl_pr, vl_Y)

                gen_trloss_tracc_toss_tacc = [(i+1), tmp_tr_loss, tmp_tr_acc, tmp_vl_loss, tmp_vl_acc]
                print('Generation # {}. Train Loss: {:.2f}. Train Acc: {:.2f}.\tTest Loss: {:.2f}. Test Acc: {:.2f}'.format(*gen_trloss_tracc_toss_tacc))