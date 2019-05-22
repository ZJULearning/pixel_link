import numpy as np

def get_neighbours(x, y):
    return [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1), \
        (x - 1, y),                 (x + 1, y),  \
        (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)]

def is_valid_cord(x, y, w, h):
    return x >=0 and x < w and y >= 0 and y < h
    
def decode_image_by_join(pixel_scores, link_scores, pixel_conf_threshold, link_conf_threshold):
    pixel_mask = pixel_scores >= pixel_conf_threshold
    link_mask = link_scores >= link_conf_threshold
    done_mask = np.zeros(pixel_mask.shape, np.bool)
    result_mask = np.zeros(pixel_mask.shape, np.int32)
    points = list(zip(*np.where(pixel_mask)))
    h, w = np.shape(pixel_mask)
    group_id = 0
    for point in points:
        if done_mask[point]:
            continue
        group_id += 1
        group_q = [point]
        result_mask[point] = group_id
        while len(group_q):
            y, x = group_q[-1]
            group_q.pop()
            if not done_mask[y,x]:
                done_mask[y,x], result_mask[y,x] = True, group_id
                for n_idx, (nx, ny) in enumerate(get_neighbours(x, y)):
                    if is_valid_cord(nx, ny, w, h) and pixel_mask[ny, nx] and (link_mask[y, x, n_idx] or link_mask[ny, nx, 7 - n_idx]):
                        group_q.append((ny, nx))
    return result_mask
