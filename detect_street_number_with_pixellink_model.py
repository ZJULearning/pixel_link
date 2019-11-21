# encoding = utf-8
"""
Input:
    * house_front_image_listing_csv, mandatory column: image_fpath,
    the path is relative to the image_base_dir below
    * image_base_dir
    * checkpoint_path, the checkpoint file of trained model
    * detection_result_output_json_path, notice it is a line json file, i.e. each line is a json obj
    * optional detection_result_image_detection_result_image_output_dir

Output:
    * create detection_result_output_json file at detection_result_output_json_path
        Notice that only images which have some street number detected will appear in this file

    * optionally output images with detected bounding boxes rendered in detection_result_image_detection_result_image_output_dir.
        Notice that only images which have some street number detected will be output here
"""
import glob
import json
import os

import numpy as np
import tensorflow as tf
import pandas as pd

import pixel_link
from preprocessing import ssd_vgg_preprocessing
from nets import pixel_link_symbol
import config
import util

slim = tf.contrib.slim


tf.app.flags.DEFINE_string(
    'house_front_image_listing_csv',
    '/home/victor/workspace/str_image_match/sample_data/mono/airbnb_house_front_image.excluding_duplicated_8_times_or_more.mono.csv',
    'csv which list all house front image files in  image_base_dir'
)
tf.app.flags.DEFINE_string(
    'image_base_dir',
    '/home/victor/workspace/str_image_match/images/airbnb_image',
    'base folder where the images files reside'
)
tf.app.flags.DEFINE_string(
    'checkpoint_path',
    'runs/checkpoint_0.879/model.ckpt-20677',
    'the path of pretrained model to be used'
)
tf.app.flags.DEFINE_string(
    'detection_result_output_json_path',
    '/home/victor/workspace/street_num_spotting/data/street_number_detection_result.mono.json',
    'the target path to create the output json'
)
tf.app.flags.DEFINE_string(
    'detection_result_image_output_dir',
    './output',
    'The directory where the output images should be saved.'
)

tf.app.flags.DEFINE_integer('eval_image_width', 1280, 'resized image width for inference')
tf.app.flags.DEFINE_integer('eval_image_height', 768, 'resized image height for inference')
tf.app.flags.DEFINE_float('pixel_conf_threshold', 0.5, 'threshold on the pixel confidence')
tf.app.flags.DEFINE_float('link_conf_threshold', 0.5, 'threshold on the link confidence')

tf.app.flags.DEFINE_float('gpu_memory_fraction', -1,
                          'the gpu memory fraction to be used. If less than 0, allow_growth = True is used.')

tf.app.flags.DEFINE_bool('using_moving_average', True, 'Whether to use ExponentionalMovingAverage')
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999, 'The decay rate of ExponentionalMovingAverage')

FLAGS = tf.app.flags.FLAGS


def config_initialization():
    # image shape and feature layers shape inference
    image_shape = (FLAGS.eval_image_height, FLAGS.eval_image_width)

    if not FLAGS.image_base_dir:
        raise ValueError('You must supply the dataset directory with --image_base_dir')

    tf.logging.set_verbosity(tf.logging.DEBUG)

    config.init_config(image_shape,
                       batch_size=1,
                       pixel_conf_threshold=FLAGS.pixel_conf_threshold,
                       link_conf_threshold=FLAGS.link_conf_threshold,
                       num_gpus=1,
                       )


def create_model():
    with tf.name_scope('evaluation_%dx%d' % (FLAGS.eval_image_height, FLAGS.eval_image_width)):
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            image = tf.placeholder(dtype=tf.int32, shape=[None, None, 3])
            processed_image, _, _, _, _ = ssd_vgg_preprocessing.preprocess_image(image, None, None, None, None,
                                                                                 out_shape=config.image_shape,
                                                                                 data_format=config.data_format,
                                                                                 is_training=False)
            b_image = tf.expand_dims(processed_image, axis=0)

            # build model and loss
            net = pixel_link_symbol.PixelLinkNet(b_image, is_training=False)
            masks = pixel_link.tf_decode_score_map_to_mask_in_batch(
                net.pixel_pos_scores, net.link_pos_scores)
    return image, masks, net


def prepare_session_config():
    sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    if FLAGS.gpu_memory_fraction < 0:
        sess_config.gpu_options.allow_growth = True
    elif FLAGS.gpu_memory_fraction > 0:
        sess_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction;


def draw_bboxes(img, bboxes, color):
    for bbox in bboxes:
        points = np.reshape(bbox, [4, 2])
        cnts = util.img.points_to_contours(points)
        util.img.draw_contours(img, contours=cnts,
                               idx=-1, color=color, border_width=1)


def list_image_in_folder_recursively(image_base_dir):
    return [os.path.relpath(f, image_base_dir)
            for f in glob.glob(image_base_dir + '/**/*.jpg', recursive=True)]


def convert_to_json(bbox):
    return {
        "label": "",
        "points": [[int(round(v[0])), int(round(v[1]))] for v in bbox.reshape([4, 2])]
    }


def save_detection_result_as_line_json(bboxes_det, image_rel_path, output_json_file):
    """

    :param bboxes_det: a list of 1D numpy array. Each array is of size 8, which gives the x,y of 4 points
    :param image_rel_path: identify the original image
    :param output_json_file: file object
    :return:
    """
    json_obj = dict(
        image_fname=image_rel_path,
        detected_numbers=[convert_to_json(bbox) for bbox in bboxes_det]
    )
    output_json_file.write(json.dumps(json_obj) + '\n')


def run_inference(global_step, image, masks, net):
    variables_to_restore = get_variables_to_restore(global_step)
    saver = tf.train.Saver(var_list=variables_to_restore)

    with tf.Session() as sess:
        saver.restore(sess, util.tf.get_latest_ckpt(FLAGS.checkpoint_path))
        tf.logging.info('model restored')

        with open(FLAGS.detection_result_output_json_path, 'w') as output_json_file:
            src_image_listing_df = pd.read_csv(FLAGS.house_front_image_listing_csv)
            total_image_num = len(src_image_listing_df)

            for i, image_rel_path in enumerate(src_image_listing_df['image_path']):
                file_path = util.io.join_path(FLAGS.image_base_dir, image_rel_path)

                image_data = util.img.imread(file_path, rgb=True)
                link_scores, pixel_scores, mask_vals = sess.run(
                    [net.link_pos_scores, net.pixel_pos_scores, masks],
                    feed_dict={image: image_data})

                image_idx = 0  # only one image in the batch
                mask = mask_vals[image_idx, ...]
                bboxes_det = pixel_link.mask_to_bboxes(mask, image_data.shape)

                if len(bboxes_det) == 0:
                    continue

                save_detection_result_as_line_json(bboxes_det, image_rel_path, output_json_file)

                if FLAGS.detection_result_image_output_dir:
                    save_image_with_rendered_bboxes(bboxes_det, image_data, image_rel_path)

                print 'processed %d/%d\r images' % (i, total_image_num),

            print('Done')


def save_image_with_rendered_bboxes(bboxes_det, image_data, image_rel_path):
    draw_bboxes(image_data, bboxes_det, util.img.COLOR_RGB_RED)
    output_img_path = os.path.join(FLAGS.detection_result_image_output_dir, image_rel_path)
    output_dir = os.path.dirname(output_img_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    util.sit(image_data, path=output_img_path)  # save image to the output file


def get_variables_to_restore(global_step):
    # Variables to restore: moving avg. or normal weights.
    if FLAGS.using_moving_average:
        variable_averages = tf.train.ExponentialMovingAverage(
            FLAGS.moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore(
            tf.trainable_variables())
        variables_to_restore[global_step.op.name] = global_step
    else:
        variables_to_restore = slim.get_variables_to_restore()
    return variables_to_restore


def main(_):
    config_initialization()

    if FLAGS.detection_result_image_output_dir:
        if os.path.exists(FLAGS.detection_result_image_output_dir):
            os.system('rm -rf %s' % FLAGS.detection_result_image_output_dir)
        os.makedirs(FLAGS.detection_result_image_output_dir)

    global_step = slim.get_or_create_global_step()
    image, masks, net = create_model()
    prepare_session_config()
    run_inference(global_step, image, masks, net)


if __name__ == '__main__':
    tf.app.run()

