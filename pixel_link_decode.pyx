import cv2
import numpy as np

import util
PIXEL_NEIGHBOUR_TYPE_4 = 'PIXEL_NEIGHBOUR_TYPE_4'
PIXEL_NEIGHBOUR_TYPE_8 = 'PIXEL_NEIGHBOUR_TYPE_8'


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
    

    
def decode_image_by_join(pixel_scores, link_scores, 
                 pixel_conf_threshold, link_conf_threshold):
    pixel_mask = pixel_scores >= pixel_conf_threshold
    link_mask = link_scores >= link_conf_threshold
    points = zip(*np.where(pixel_mask))
    h, w = np.shape(pixel_mask)
    group_mask = dict.fromkeys(points, -1)
    def find_parent(point):
        return group_mask[point]
        
    def set_parent(point, parent):
        group_mask[point] = parent
        
    def is_root(point):
        return find_parent(point) == -1
    
    def find_root(point):
        root = point
        update_parent = False
        while not is_root(root):
            root = find_parent(root)
            update_parent = True
        
        # for acceleration of find_root
        if update_parent:
            set_parent(point, root)
            
        return root
        
    def join(p1, p2):
        root1 = find_root(p1)
        root2 = find_root(p2)
        
        if root1 != root2:
            set_parent(root1, root2)
        
    def get_all():
        root_map = {}
        def get_index(root):
            if root not in root_map:
                root_map[root] = len(root_map) + 1
            return root_map[root]
        
        mask = np.zeros_like(pixel_mask, dtype = np.int32)
        for point in points:
            point_root = find_root(point)
            bbox_idx = get_index(point_root)
            mask[point] = bbox_idx
        return mask
    
    # join by link
    for point in points:
        y, x = point
        neighbours = get_neighbours(x, y)
        for n_idx, (nx, ny) in enumerate(neighbours):
            if is_valid_cord(nx, ny, w, h):
#                 reversed_neighbours = get_neighbours(nx, ny)
#                 reversed_idx = reversed_neighbours.index((x, y))
                link_value = link_mask[y, x, n_idx]# and link_mask[ny, nx, reversed_idx]
                pixel_cls = pixel_mask[ny, nx]
                if link_value and pixel_cls:
                    join(point, (ny, nx))
    
    mask = get_all()
    return mask

