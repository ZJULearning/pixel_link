# encoding=utf-8
import argparse
import os

import numpy as np
import tensorflow as tf
import util
from dataset_utils import int64_feature, float_feature, bytes_feature, convert_to_example
import config


def cvt_to_tfrecords(output_path, data_path, gt_path):
    image_names = util.io.ls(data_path, '.jpg')  # [0:10]
    print "%d images found in %s" % (len(image_names), data_path)
    with tf.python_io.TFRecordWriter(output_path) as tfrecord_writer:
        for idx, image_name in enumerate(image_names):
            oriented_bboxes = []
            bboxes = []
            labels = []  # a mask for labels_text list to indicate whether it is a 'ignored' label text or not
            labels_text = []
            path = util.io.join_path(data_path, image_name)
            print "\tconverting image: %d/%d %s" % (idx, len(image_names), image_name)
            image_data = tf.gfile.FastGFile(path, 'r').read()

            image = util.img.imread(path, rgb=True)
            shape = image.shape
            h, w = shape[0:2]
            h *= 1.0
            w *= 1.0
            image_name = util.str.split(image_name, '.')[0]
            gt_name = 'gt_' + image_name + '.txt'
            gt_filepath = util.io.join_path(gt_path, gt_name)
            lines = util.io.read_lines(gt_filepath)

            for line in lines:
                line = util.str.remove_all(line, '\xef\xbb\xbf')
                gt = util.str.split(line, ',')
                oriented_box = [int(gt[i]) for i in range(8)]
                oriented_box = np.asarray(oriented_box) / ([w, h] * 4)
                oriented_bboxes.append(oriented_box)

                xs = oriented_box.reshape(4, 2)[:, 0]
                ys = oriented_box.reshape(4, 2)[:, 1]
                xmin = xs.min()
                xmax = xs.max()
                ymin = ys.min()
                ymax = ys.max()
                bboxes.append([xmin, ymin, xmax, ymax])

                # might be wrong here, but it doesn't matter because the label is not going to be used in detection
                labels_text.append(gt[-1])
                ignored = util.str.contains(gt[-1], '###')
                if ignored:
                    labels.append(config.ignore_label)
                else:
                    labels.append(config.text_label)
            example = convert_to_example(image_data, image_name, labels, labels_text, bboxes, oriented_bboxes, shape)
            tfrecord_writer.write(example.SerializeToString())


def main(args):
    output_dir = os.path.dirname(args.output_tfrecords_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cvt_to_tfrecords(output_path=args.output_tfrecords_path, data_path=args.image_dir, gt_path=args.label_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_dir", default='/home/victor/workspace/datasets/scene_text/street_number_recognition/train')
    parser.add_argument("-l", "--label_dir", default='/home/victor/workspace/datasets/scene_text/street_number_recognition/train_ic15_format_label')
    parser.add_argument("-t", "--output_tfrecords_path", default='/home/victor/workspace/datasets/scene_text/street_number_recognition/train.tfrecord')
    args = parser.parse_args()
    main(args)

    # root_dir = util.io.get_absolute_path('~/workspace/datasets/scene_text/ICDAR2015/detection/')
    # output_dir = util.io.get_absolute_path('~/workspace/pixel_link/tfrecord/icdar2015/')
    # util.io.mkdir(output_dir)
    # training_data_dir = util.io.join_path(root_dir, 'ch4_training_images')
    # training_gt_dir = util.io.join_path(root_dir, 'ch4_training_localization_transcription_gt')
    #
    # test_data_dir = util.io.join_path(root_dir, 'ch4_test_images')
    # test_gt_dir = util.io.join_path(root_dir, 'ch4_test_localization_transcription_gt')
    # cvt_to_tfrecords(output_path=util.io.join_path(output_dir, 'icdar2015_test.tfrecord'), data_path=test_data_dir,
    #                  gt_path=test_gt_dir)
    #


