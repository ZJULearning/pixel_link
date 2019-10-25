from __future__ import print_function
from pprint import pprint
import numpy as np
from tensorflow.contrib.slim.python.slim.data import parallel_reader
import tensorflow as tf
import util
from nets import pixel_link_symbol
import pixel_link
slim = tf.contrib.slim

#=====================================================================
#====================Pre-processing params START======================
# VGG mean parameters.
r_mean = 123.
g_mean = 117.
b_mean = 104.
rgb_mean = [r_mean, g_mean, b_mean]

# scale, crop, filtering and resize parameters
use_rotation = True
rotation_prob = 0.5
max_expand_scale = 1
expand_prob = 0
min_object_covered = 0.1          # Minimum object to be cropped in random crop.
bbox_crop_overlap = 0.2         # Minimum overlap to keep a bbox after cropping.
crop_aspect_ratio_range = (0.5, 2.)  # Distortion ratio during cropping.
area_range = [0.1, 1]
flip = False
using_shorter_side_filtering=True
min_shorter_side = 6
max_shorter_side = np.infty
#====================Pre-processing params END========================
#=====================================================================




#=====================================================================
#====================Post-processing params START=====================
decode_method = pixel_link.DECODE_METHOD_join
min_area = 100
min_height = 6
#====================Post-processing params END=======================
#=====================================================================



#=====================================================================
#====================Training and model params START =================
dropout_ratio = 0
max_neg_pos_ratio = 3

feat_fuse_type = pixel_link_symbol.FUSE_TYPE_cascade_conv1x1_upsample_sum
# feat_fuse_type = pixel_link_symbol.FUSE_TYPE_cascade_conv1x1_128_upsamle_sum_conv1x1_2
# feat_fuse_type = pixel_link_symbol.FUSE_TYPE_cascade_conv1x1_128_upsamle_concat_conv1x1_2

pixel_neighbour_type = pixel_link.PIXEL_NEIGHBOUR_TYPE_8
#pixel_neighbour_type = pixel_link.PIXEL_NEIGHBOUR_TYPE_4


#model_type = pixel_link_symbol.MODEL_TYPE_vgg16
#feat_layers = ['conv2_2', 'conv3_3', 'conv4_3', 'conv5_3', 'fc7']
#strides = [2]
model_type = pixel_link_symbol.MODEL_TYPE_vgg16
feat_layers = ['conv3_3', 'conv4_3', 'conv5_3', 'fc7']
strides = [4]

pixel_cls_weight_method = pixel_link.PIXEL_CLS_WEIGHT_bbox_balanced
bbox_border_width = 1
pixel_cls_border_weight_lambda = 1.0
pixel_cls_loss_weight_lambda = 2.0
pixel_link_neg_loss_weight_lambda = 1.0
pixel_link_loss_weight = 1.0
#====================Training and model params END ==================
#=====================================================================


#=====================================================================
#====================do-not-change configurations START===============
num_classes = 2
ignore_label = -1
background_label = 0
text_label = 1
data_format = 'NHWC'
train_with_ignored = False
#====================do-not-change configurations END=================
#=====================================================================

global weight_decay

global train_image_shape
global image_shape
global score_map_shape

global batch_size
global batch_size_per_gpu
global gpus
global num_clones
global clone_scopes

global num_neighbours

global pixel_conf_threshold
global link_conf_threshold

def _set_weight_decay(wd):
    global weight_decay
    weight_decay = wd

def _set_image_shape(shape):
    h, w = shape
    global train_image_shape
    global score_map_shape
    global image_shape
    
    assert w % 4 == 0
    assert h % 4 == 0
    
    train_image_shape = [h, w]
    score_map_shape = (h / strides[0], w / strides[0])
    image_shape = train_image_shape

def _set_batch_size(bz):
    global batch_size
    batch_size = bz

def _set_seg_th(pixel_conf_th, link_conf_th):
    global pixel_conf_threshold
    global link_conf_threshold
    
    pixel_conf_threshold = pixel_conf_th
    link_conf_threshold = link_conf_th
    
    
def  _set_train_with_ignored(train_with_ignored_):
    global train_with_ignored    
    train_with_ignored = train_with_ignored_

    
def init_config(image_shape, batch_size = 1, 
                weight_decay = 0.0005, 
                num_gpus = 1, 
                pixel_conf_threshold = 0.6,
                link_conf_threshold = 0.9):
    _set_seg_th(pixel_conf_threshold, link_conf_threshold)
    _set_weight_decay(weight_decay)
    _set_image_shape(image_shape)

    #init batch size
    global gpus
    gpus = util.tf.get_available_gpus(num_gpus)
    
    global num_clones
    num_clones = len(gpus)
    
    global clone_scopes
    clone_scopes = ['clone_%d'%(idx) for idx in xrange(num_clones)]
    
    _set_batch_size(batch_size)
    
    global batch_size_per_gpu
    batch_size_per_gpu = batch_size / num_clones
    if batch_size_per_gpu < 1:
        raise ValueError('Invalid batch_size [=%d], \
                resulting in 0 images per gpu.'%(batch_size))
    
    global num_neighbours
    num_neighbours = pixel_link.get_neighbours_fn()[1]

    
def print_config(flags, dataset, save_dir = None, print_to_file = True):
    def do_print(stream=None):
        print(util.log.get_date_str(), file = stream)
        print('\n# =========================================================================== #', file=stream)
        print('# Training flags:', file=stream)
        print('# =========================================================================== #', file=stream)
        
        def print_ckpt(path):
            ckpt = util.tf.get_latest_ckpt(path)
            if ckpt is not None:
                print('Resume Training from : %s'%(ckpt), file = stream)
                return True
            return False
        
        if not print_ckpt(flags.train_dir):
            print_ckpt(flags.checkpoint_path)                
            
        pprint(flags.__flags, stream=stream)

        print('\n# =========================================================================== #', file=stream)
        print('# pixel_link net parameters:', file=stream)
        print('# =========================================================================== #', file=stream)
        vars = globals()
        for key in vars:
            var = vars[key]
            if util.dtype.is_number(var) or util.dtype.is_str(var) or util.dtype.is_list(var) or util.dtype.is_tuple(var):
                pprint('%s=%s'%(key, str(var)), stream = stream)
            
        print('\n# =========================================================================== #', file=stream)
        print('# Training | Evaluation dataset files:', file=stream)
        print('# =========================================================================== #', file=stream)
        data_files = parallel_reader.get_data_files(dataset.data_sources)
        pprint(sorted(data_files), stream=stream)
        print('', file=stream)
    do_print(None)
    
    if print_to_file:
        # Save to a text file as well.
        if save_dir is None:
            save_dir = flags.train_dir
            
        util.io.mkdir(save_dir)
        path = util.io.join_path(save_dir, 'training_config.txt')
        with open(path, "a") as out:
            do_print(out)
    
def load_config(path):
    if not util.io.is_dir(path):
        path = util.io.get_dir(path)
        
    config_file = util.io.join_path(path, 'config.py')
    if util.io.exists(config_file):
        tf.logging.info('loading config.py from %s'%(config_file))
        config = util.mod.load_mod_from_path(config_file)
    else:
        util.io.copy('config.py', path)
