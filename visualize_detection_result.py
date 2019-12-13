#encoding utf-8

import numpy as np
import util


def draw_bbox(image_data, line, color):
    line = util.str.remove_all(line, '\xef\xbb\xbf')
    data = line.split(',');
    points = [int(v) for v in data[0:8]]
    points = np.reshape(points, (4, 2))
    cnts = util.img.points_to_contours(points)
    util.img.draw_contours(image_data, cnts, -1, color = color, border_width = 3)
    
       
def visualize(image_root, det_root, output_root, gt_root = None):
    def read_gt_file(image_name):
        gt_file = util.io.join_path(gt_root, 'gt_%s.txt'%(image_name))
        return util.io.read_lines(gt_file)

    def read_det_file(image_name):
        det_file = util.io.join_path(det_root, 'res_%s.txt'%(image_name))
        return util.io.read_lines(det_file)
    
    def read_image_file(image_name):
        return util.img.imread(util.io.join_path(image_root, image_name))
    
    image_names = util.io.ls(image_root, '.jpg')
    for image_idx, image_name in enumerate(image_names):
        print '%d / %d: %s'%(image_idx + 1, len(image_names), image_name)
        image_data = read_image_file(image_name) # in BGR
        image_name = image_name.split('.')[0]
        if det_root is not None:
            det_image = image_data.copy()
            det_lines = read_det_file(image_name)
            for line in det_lines:
                draw_bbox(det_image, line, color = util.img.COLOR_GREEN)
            output_path = util.io.join_path(output_root, '%s_pred.jpg'%(image_name))
            util.img.imwrite(output_path, det_image)
            print "Detection result has been written to ", util.io.get_absolute_path(output_path)
        
        if gt_root is not None:
            gt_lines = read_gt_file(image_name)
            for line in gt_lines:
                draw_bbox(image_data, line, color = util.img.COLOR_GREEN)
            util.img.imwrite(util.io.join_path(output_root, '%s_gt.jpg'%(image_name)), image_data)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='visualize detection result of pixel_link')
    parser.add_argument('--image', type=str, required = True,help='the directory of test image')
    parser.add_argument('--gt', type=str, default=None, help='the directory of ground truth txt files')
    parser.add_argument('--det', type=str, default=None, help='the directory of detection result')
    parser.add_argument('--output', type=str, required = True, help='the directory to store images with bboxes')
    
    args = parser.parse_args()

    print('**************Arguments*****************')
    print(args)
    print('****************************************')

    if not args.det and not args.gt:
        print("At least one of --gt or --det should be specified")
    else:
        visualize(image_root = args.image, gt_root = args.gt, det_root = args.det, output_root = args.output)
