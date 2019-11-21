import tensorflow as tf
import numpy as np
import cv2

import util

PIXEL_CLS_WEIGHT_all_ones = 'PIXEL_CLS_WEIGHT_all_ones' 
PIXEL_CLS_WEIGHT_bbox_balanced = 'PIXEL_CLS_WEIGHT_bbox_balanced'
PIXEL_NEIGHBOUR_TYPE_4 = 'PIXEL_NEIGHBOUR_TYPE_4'
PIXEL_NEIGHBOUR_TYPE_8 = 'PIXEL_NEIGHBOUR_TYPE_8'

DECODE_METHOD_join = 'DECODE_METHOD_join'


def get_neighbours_8(x, y):
    """
    Get 8 neighbours of point(x, y)
    """
    return [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1), \
        (x - 1, y),                 (x + 1, y),  \
        (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)]


def get_neighbours_4(x, y):
    return [(x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1)]


def get_neighbours(x, y):
    import config
    neighbour_type = config.pixel_neighbour_type
    if neighbour_type == PIXEL_NEIGHBOUR_TYPE_4:
        return get_neighbours_4(x, y)
    else:
        return get_neighbours_8(x, y)
    
def get_neighbours_fn():
    import config
    neighbour_type = config.pixel_neighbour_type
    if neighbour_type == PIXEL_NEIGHBOUR_TYPE_4:
        return get_neighbours_4, 4
    else:
        return get_neighbours_8, 8



def is_valid_cord(x, y, w, h):
    """
    Tell whether the 2D coordinate (x, y) is valid or not.
    If valid, it should be on an h x w image
    """
    return x >=0 and x < w and y >= 0 and y < h;

#=====================Ground Truth Calculation Begin==================
def tf_cal_gt_for_single_image(xs, ys, labels):
    pixel_cls_label, pixel_cls_weight,  \
    pixel_link_label, pixel_link_weight = \
        tf.py_func(
                    cal_gt_for_single_image, 
                    [xs, ys, labels],
                    [tf.int32, tf.float32, tf.int32, tf.float32]
                   )
    import config
    score_map_shape = config.score_map_shape
    num_neighbours = config.num_neighbours
    h, w = score_map_shape
    pixel_cls_label.set_shape(score_map_shape)
    pixel_cls_weight.set_shape(score_map_shape)
    pixel_link_label.set_shape([h, w, num_neighbours])
    pixel_link_weight.set_shape([h, w, num_neighbours])
    return pixel_cls_label, pixel_cls_weight, \
            pixel_link_label, pixel_link_weight


def cal_gt_for_single_image(normed_xs, normed_ys, labels):
    """
    Args:
        xs, ys: both in shape of (N, 4), 
            and N is the number of bboxes,
            their values are normalized to [0,1]
        labels: shape = (N,), only two values are allowed:
                                                        -1: ignored
                                                        1: text
    Return:
        pixel_cls_label
        pixel_cls_weight
        pixel_link_label
        pixel_link_weight
    """
    import config
    score_map_shape = config.score_map_shape
    pixel_cls_weight_method  = config.pixel_cls_weight_method
    h, w = score_map_shape
    text_label = config.text_label
    ignore_label = config.ignore_label
    background_label = config.background_label
    num_neighbours = config.num_neighbours
    bbox_border_width = config.bbox_border_width
    pixel_cls_border_weight_lambda = config.pixel_cls_border_weight_lambda
    
    # validate the args
    assert np.ndim(normed_xs) == 2
    assert np.shape(normed_xs)[-1] == 4
    assert np.shape(normed_xs) == np.shape(normed_ys)
    assert len(normed_xs) == len(labels)
    
#     assert set(labels).issubset(set([text_label, ignore_label, background_label]))

    num_positive_bboxes = np.sum(np.asarray(labels) == text_label)
    # rescale normalized xys to absolute values
    xs = normed_xs * w
    ys = normed_ys * h
    
    # initialize ground truth values
    mask = np.zeros(score_map_shape, dtype = np.int32)
    pixel_cls_label = np.ones(score_map_shape, dtype = np.int32) * background_label
    pixel_cls_weight = np.zeros(score_map_shape, dtype = np.float32)
    
    pixel_link_label = np.zeros((h, w, num_neighbours), dtype = np.int32)
    pixel_link_weight = np.ones((h, w, num_neighbours), dtype = np.float32)
    
    # find overlapped pixels, and consider them as ignored in pixel_cls_weight
    # and pixels in ignored bboxes are ignored as well
    # That is to say, only the weights of not ignored pixels are set to 1
    
    ## get the masks of all bboxes
    bbox_masks = []
    pos_mask = mask.copy()
    for bbox_idx, (bbox_xs, bbox_ys) in enumerate(zip(xs, ys)):
        if labels[bbox_idx] == background_label:
            continue
        
        bbox_mask = mask.copy()
        
        bbox_points = zip(bbox_xs, bbox_ys)
        bbox_contours = util.img.points_to_contours(bbox_points)
        # border_width = -1 means fill the interior
        util.img.draw_contours(bbox_mask, bbox_contours, idx = -1, 
                               color = 1, border_width = -1)
        
        bbox_masks.append(bbox_mask)
        
        if labels[bbox_idx] == text_label:
            pos_mask += bbox_mask  # basically pos_mask contains all individual bounding boxes
        
    # treat overlapped in-bbox pixels as negative, 
    # and non-overlapped  ones as positive
    pos_mask = np.asarray(pos_mask == 1, dtype = np.int32)
    num_positive_pixels = np.sum(pos_mask)
    
    ## add all bbox_maskes, find non-overlapping pixels
    sum_mask = np.sum(bbox_masks, axis = 0)
    not_overlapped_mask = sum_mask == 1
    #TBD: isn't not_overlapped_mask the same as pos_mask?
    
    
    ## gt and weight calculation
    for bbox_idx, bbox_mask in enumerate(bbox_masks):
        bbox_label = labels[bbox_idx]
        if bbox_label == ignore_label:
            # for ignored bboxes, only non-overlapped pixels are encoded as ignored 
            bbox_ignore_pixel_mask = bbox_mask * not_overlapped_mask
            pixel_cls_label += bbox_ignore_pixel_mask * ignore_label
            continue
        
        if labels[bbox_idx] == background_label:
            continue
        # from here on, only text boxes left.
        
        # for positive bboxes, all pixels within it and pos_mask are positive
        bbox_positive_pixel_mask = bbox_mask * pos_mask
        # background or text is encoded into cls gt
        pixel_cls_label += bbox_positive_pixel_mask * bbox_label

        # for the pixel cls weights, only positive pixels are set to ones
        if pixel_cls_weight_method == PIXEL_CLS_WEIGHT_all_ones:
            pixel_cls_weight += bbox_positive_pixel_mask
        elif pixel_cls_weight_method == PIXEL_CLS_WEIGHT_bbox_balanced:
            # let N denote num_positive_pixels
            # weight per pixel = N /num_positive_bboxes / n_pixels_in_bbox
            # so all pixel weights in this bbox sum to N/num_positive_bboxes
            # and all pixels weights in this image sum to N, the same
            # as setting all weights to 1
            num_bbox_pixels = np.sum(bbox_positive_pixel_mask)
            if num_bbox_pixels > 0: 
                per_bbox_weight = num_positive_pixels * 1.0 / num_positive_bboxes
                per_pixel_weight = per_bbox_weight / num_bbox_pixels
                pixel_cls_weight += bbox_positive_pixel_mask * per_pixel_weight
        else:
            raise ValueError, 'pixel_cls_weight_method not supported:%s'\
                        %(pixel_cls_weight_method)

    
        ## calculate the labels and weights of links
        ### for all pixels in  bboxes, all links are positive at first
        bbox_point_cords = np.where(bbox_positive_pixel_mask)
        pixel_link_label[bbox_point_cords] = 1


        ## the border of bboxes might be distored because of overlapping
        ## so recalculate it, and find the border mask        
        new_bbox_contours = util.img.find_contours(bbox_positive_pixel_mask)
        bbox_border_mask = mask.copy()
        util.img.draw_contours(bbox_border_mask, new_bbox_contours, -1, 
                   color = 1, border_width = bbox_border_width * 2 + 1)
        bbox_border_mask *= bbox_positive_pixel_mask
        bbox_border_cords = np.where(bbox_border_mask)

        ## give more weight to the border pixels if configured
        pixel_cls_weight[bbox_border_cords] *= pixel_cls_border_weight_lambda
        
        ### change link labels according to their neighbour status
        border_points = zip(*bbox_border_cords)
        def in_bbox(nx, ny):
            return bbox_positive_pixel_mask[ny, nx]
        
        for y, x in border_points:
            neighbours = get_neighbours(x, y)
            for n_idx, (nx, ny) in enumerate(neighbours):
                if not is_valid_cord(nx, ny, w, h) or not in_bbox(nx, ny):
                    pixel_link_label[y, x, n_idx] = 0

    pixel_cls_weight = np.asarray(pixel_cls_weight, dtype = np.float32)    
    pixel_link_weight *= np.expand_dims(pixel_cls_weight, axis = -1)

#     try:
#         np.testing.assert_almost_equal(np.sum(pixel_cls_weight), num_positive_pixels, decimal = 1)
#     except:
#         print  num_positive_pixels, np.sum(pixel_cls_label), np.sum(pixel_cls_weight)
#         import pdb
#         pdb.set_trace()
    return pixel_cls_label, pixel_cls_weight, pixel_link_label, pixel_link_weight

#=====================Ground Truth Calculation End====================


#============================Decode Begin=============================

def tf_decode_score_map_to_mask_in_batch(pixel_cls_scores, pixel_link_scores):
    masks = tf.py_func(decode_batch, 
                       [pixel_cls_scores, pixel_link_scores], tf.int32)
    b, h, w = pixel_cls_scores.shape.as_list()
    masks.set_shape([b, h, w])
    return masks

    

def decode_batch(pixel_cls_scores, pixel_link_scores, 
                 pixel_conf_threshold = None, link_conf_threshold = None):
    import config
    
    if pixel_conf_threshold is None:
        pixel_conf_threshold = config.pixel_conf_threshold
    
    if link_conf_threshold is None:
        link_conf_threshold = config.link_conf_threshold
    
    batch_size = pixel_cls_scores.shape[0]
    batch_mask = []
    for image_idx in xrange(batch_size):
        image_pos_pixel_scores = pixel_cls_scores[image_idx, :, :]
        image_pos_link_scores = pixel_link_scores[image_idx, :, :, :]    
        mask = decode_image(
            image_pos_pixel_scores, image_pos_link_scores, 
            pixel_conf_threshold, link_conf_threshold
        )
        batch_mask.append(mask)
    return np.asarray(batch_mask, np.int32)

# @util.dec.print_calling_in_short
# @util.dec.timeit
def decode_image(pixel_scores, link_scores, 
                 pixel_conf_threshold, link_conf_threshold):
    import config
    if config.decode_method == DECODE_METHOD_join:
        mask =  decode_image_by_join(pixel_scores, link_scores, 
                 pixel_conf_threshold, link_conf_threshold)
        return mask
    elif config.decode_method == DECODE_METHOD_border_split:
        return decode_image_by_border(pixel_scores, link_scores, 
                 pixel_conf_threshold, link_conf_threshold)
    else:
        raise ValueError('Unknow decode method:%s'%(config.decode_method))


import pyximport; pyximport.install()    
from pixel_link_decode import decode_image_by_join

def min_area_rect(cnt):
    """
    Args:
        xs: numpy ndarray with shape=(N,4). N is the number of oriented bboxes. 4 contains [x1, x2, x3, x4]
        ys: numpy ndarray with shape=(N,4), [y1, y2, y3, y4]
            Note that [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] can represent an oriented bbox.
    Return:
        the oriented rects sorrounding the box, in the format:[cx, cy, w, h, theta]. 
    """
    rect = cv2.minAreaRect(cnt)
    cx, cy = rect[0]
    w, h = rect[1]
    theta = rect[2]
    box = [cx, cy, w, h, theta]
    return box, w * h

def rect_to_xys(rect, image_shape):
    """Convert rect to xys, i.e., eight points
    The `image_shape` is used to to make sure all points return are valid, i.e., within image area
    """
    h, w = image_shape[0:2]
    def get_valid_x(x):
        if x < 0:
            return 0
        if x >= w:
            return w - 1
        return x
    
    def get_valid_y(y):
        if y < 0:
            return 0
        if y >= h:
            return h - 1
        return y
    
    rect = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4])
    points = cv2.cv.BoxPoints(rect)
    points = np.int0(points)
    for i_xy, (x, y) in enumerate(points):
        x = get_valid_x(x)
        y = get_valid_y(y)
        points[i_xy, :] = [x, y]
    points = np.reshape(points, -1)
    return points

# @util.dec.print_calling_in_short
# @util.dec.timeit
def mask_to_bboxes(mask, image_shape =  None, min_area = None, 
                   min_height = None, min_aspect_ratio = None):
    import config
    feed_shape = config.train_image_shape
    
    if image_shape is None:
        image_shape = feed_shape
        
    image_h, image_w = image_shape[0:2]
    
    if min_area is None:
        min_area = config.min_area
        
    if min_height is None:
        min_height = config.min_height
    bboxes = []
    max_bbox_idx = mask.max()
    mask = util.img.resize(img = mask, size = (image_w, image_h), 
                           interpolation = cv2.INTER_NEAREST)
    
    for bbox_idx in xrange(1, max_bbox_idx + 1):
        bbox_mask = mask == bbox_idx
#         if bbox_mask.sum() < 10:
#             continue
        cnts = util.img.find_contours(bbox_mask)
        if len(cnts) == 0:
            continue
        cnt = cnts[0]
        rect, rect_area = min_area_rect(cnt)

        # rect is [cx, cy, w, h, theta]
        w, h = rect[2:-1]
        if min(w, h) < min_height:
            continue
        
        if rect_area < min_area:
            continue
        
#         if max(w, h) * 1.0 / min(w, h) < 2:
#             continue
        xys = rect_to_xys(rect, image_shape)
        bboxes.append(xys)
        
    return bboxes


#============================Decode End===============================
