#test code to make sure the ground truth calculation and data batch works well.

import numpy as np
import tensorflow as tf # test
from tensorflow.python.ops import control_flow_ops

from datasets import dataset_factory

from nets import pixel_link_symbol
import util
import pixel_link

slim = tf.contrib.slim
import config
# =========================================================================== #
# Checkpoint and running Flags
# =========================================================================== #
tf.app.flags.DEFINE_string('train_dir', None, 
                           'the path to store checkpoints and eventfiles for summaries')

tf.app.flags.DEFINE_string('checkpoint_path', None, 
   'the path of pretrained model to be used. If there are checkpoints in train_dir, this config will be ignored.')

tf.app.flags.DEFINE_float('gpu_memory_fraction', -1, 
                          'the gpu memory fraction to be used. If less than 0, allow_growth = True is used.')

tf.app.flags.DEFINE_integer('batch_size', None, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer('num_gpus', 1, 'The number of gpus can be used.')
tf.app.flags.DEFINE_integer('max_number_of_steps', 1000000, 'The maximum number of training steps.')
tf.app.flags.DEFINE_integer('log_every_n_steps', 1, 'log frequency')
tf.app.flags.DEFINE_bool("ignore_missing_vars", False, '')
tf.app.flags.DEFINE_string('checkpoint_exclude_scopes', None, 'checkpoint_exclude_scopes')

# =========================================================================== #
# Optimizer configs.
# =========================================================================== #
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate.')
tf.app.flags.DEFINE_float('momentum', 0.9, 'The momentum for the MomentumOptimizer')
tf.app.flags.DEFINE_float('weight_decay', 0.0001, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_bool('using_moving_average', True, 'Whether to use ExponentionalMovingAverage')
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999, 'The decay rate of ExponentionalMovingAverage')

# =========================================================================== #
# I/O and preprocessing Flags.
# =========================================================================== #
tf.app.flags.DEFINE_integer(
    'num_readers', 1,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 24,
    'The number of threads used to create the batches.')

# =========================================================================== #
# Dataset Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'dataset_name', None, 'The name of the dataset to load.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_integer('train_image_width', 512, 'Train image size')
tf.app.flags.DEFINE_integer('train_image_height', 512, 'Train image size')


FLAGS = tf.app.flags.FLAGS
def config_initialization():
    # image shape and feature layers shape inference
    image_shape = (FLAGS.train_image_height, FLAGS.train_image_width)
    
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    
    tf.logging.set_verbosity(tf.logging.DEBUG)
    util.init_logger(
        log_file = 'log_train_pixel_link_%d_%d.log'%image_shape, 
                    log_path = FLAGS.train_dir, stdout = False, mode = 'a')
    
    
    config.load_config(FLAGS.train_dir)
            
    config.init_config(image_shape, 
                       batch_size = FLAGS.batch_size, 
                       weight_decay = FLAGS.weight_decay, 
                       num_gpus = FLAGS.num_gpus
                   )

    batch_size = config.batch_size
    batch_size_per_gpu = config.batch_size_per_gpu
        
    tf.summary.scalar('batch_size', batch_size)
    tf.summary.scalar('batch_size_per_gpu', batch_size_per_gpu)

    util.proc.set_proc_name('train_pixel_link_on'+ '_' + FLAGS.dataset_name)
    
    dataset = dataset_factory.get_dataset(FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)
    config.print_config(FLAGS, dataset)
    return dataset

def create_dataset_batch_queue(dataset):
    from preprocessing import ssd_vgg_preprocessing

    with tf.device('/cpu:0'):
        with tf.name_scope(FLAGS.dataset_name + '_data_provider'):
            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers=FLAGS.num_readers,
                common_queue_capacity=1000 * config.batch_size,
                common_queue_min=700 * config.batch_size,
                shuffle=True)
        # Get for SSD network: image, labels, bboxes.
        [image, glabel, gbboxes, x1, x2, x3, x4, y1, y2, y3, y4] = provider.get([
                                                         'image',
                                                         'object/label',
                                                         'object/bbox', 
                                                         'object/oriented_bbox/x1',
                                                         'object/oriented_bbox/x2',
                                                         'object/oriented_bbox/x3',
                                                         'object/oriented_bbox/x4',
                                                         'object/oriented_bbox/y1',
                                                         'object/oriented_bbox/y2',
                                                         'object/oriented_bbox/y3',
                                                         'object/oriented_bbox/y4'
                                                         ])
        gxs = tf.transpose(tf.stack([x1, x2, x3, x4])) #shape = (N, 4)
        gys = tf.transpose(tf.stack([y1, y2, y3, y4]))
        image = tf.identity(image, 'input_image')
        
        # Pre-processing image, labels and bboxes.
        image, glabel, gbboxes, gxs, gys = \
                ssd_vgg_preprocessing.preprocess_image(
                       image, glabel, gbboxes, gxs, gys, 
                       out_shape = config.train_image_shape,
                       data_format = config.data_format, 
                       use_rotation = config.use_rotation,
                       is_training = True)
        image = tf.identity(image, 'processed_image')
        
        # calculate ground truth
        pixel_cls_label, pixel_cls_weight, \
        pixel_link_label, pixel_link_weight = \
            pixel_link.tf_cal_gt_for_single_image(gxs, gys, glabel)
        
        # batch them
        with tf.name_scope(FLAGS.dataset_name + '_batch'):
            b_image, b_pixel_cls_label, b_pixel_cls_weight, \
            b_pixel_link_label, b_pixel_link_weight = \
                tf.train.batch(
                    [image, pixel_cls_label, pixel_cls_weight, 
                        pixel_link_label, pixel_link_weight],
                    batch_size = config.batch_size_per_gpu,
                    num_threads= FLAGS.num_preprocessing_threads,
                    capacity = 500)
        with tf.name_scope(FLAGS.dataset_name + '_prefetch_queue'):
            batch_queue = slim.prefetch_queue.prefetch_queue(
                [b_image, b_pixel_cls_label, b_pixel_cls_weight, 
                    b_pixel_link_label, b_pixel_link_weight],
                capacity = 50) 
    return batch_queue    

def sum_gradients(clone_grads):                        
    averaged_grads = []
    for grad_and_vars in zip(*clone_grads):
        grads = []
        var = grad_and_vars[0][1]
        try:
            for g, v in grad_and_vars:
                assert v == var
                grads.append(g)
            grad = tf.add_n(grads, name = v.op.name + '_summed_gradients')
        except:
            import pdb
            pdb.set_trace()
        
        averaged_grads.append((grad, v))
        
#         tf.summary.histogram("variables_and_gradients_" + grad.op.name, grad)
#         tf.summary.histogram("variables_and_gradients_" + v.op.name, v)
#         tf.summary.scalar("variables_and_gradients_" + grad.op.name+\
#               '_mean/var_mean', tf.reduce_mean(grad)/tf.reduce_mean(var))
#         tf.summary.scalar("variables_and_gradients_" + v.op.name+'_mean',tf.reduce_mean(var))
    return averaged_grads


def create_clones(batch_queue):        
    with tf.device('/cpu:0'):
        global_step = slim.create_global_step()
        learning_rate = tf.constant(FLAGS.learning_rate, name='learning_rate')
        optimizer = tf.train.MomentumOptimizer(learning_rate, 
                               momentum=FLAGS.momentum, name='Momentum')

        tf.summary.scalar('learning_rate', learning_rate)
    # place clones
    pixel_link_loss = 0; # for summary only
    gradients = []
    for clone_idx, gpu in enumerate(config.gpus):
        do_summary = clone_idx == 0 # only summary on the first clone
        reuse = clone_idx > 0
        with tf.variable_scope(tf.get_variable_scope(), reuse = reuse):
            with tf.name_scope(config.clone_scopes[clone_idx]) as clone_scope:
                with tf.device(gpu) as clone_device:
                    b_image, b_pixel_cls_label, b_pixel_cls_weight, \
                        b_pixel_link_label, b_pixel_link_weight = batch_queue.dequeue()
                    # build model and loss
                    net = pixel_link_symbol.PixelLinkNet(b_image, is_training = True)
                    net.build_loss(
                        pixel_cls_labels = b_pixel_cls_label, 
                        pixel_cls_weights = b_pixel_cls_weight, 
                        pixel_link_labels = b_pixel_link_label, 
                        pixel_link_weights = b_pixel_link_weight,
                        do_summary = do_summary)
                    
                    # gather losses
                    losses = tf.get_collection(tf.GraphKeys.LOSSES, clone_scope)
                    assert len(losses) ==  2
                    total_clone_loss = tf.add_n(losses) / config.num_clones
                    pixel_link_loss += total_clone_loss

                    # gather regularization loss and add to clone_0 only
                    if clone_idx == 0:
                        regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                        total_clone_loss = total_clone_loss + regularization_loss
                    
                    # compute clone gradients
                    clone_gradients = optimizer.compute_gradients(total_clone_loss)
                    gradients.append(clone_gradients)
                    
    tf.summary.scalar('pixel_link_loss', pixel_link_loss)
    tf.summary.scalar('regularization_loss', regularization_loss)
    
    # add all gradients together
    # note that the gradients do not need to be averaged, because the average operation has been done on loss.
    averaged_gradients = sum_gradients(gradients)
    
    apply_grad_op = optimizer.apply_gradients(averaged_gradients, global_step=global_step)
    
    train_ops = [apply_grad_op]
    
    bn_update_op = util.tf.get_update_op()
    if bn_update_op is not None:
        train_ops.append(bn_update_op)
    
    # moving average
    if FLAGS.using_moving_average:
        tf.logging.info('using moving average in training, \
        with decay = %f'%(FLAGS.moving_average_decay))
        ema = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay)
        ema_op = ema.apply(tf.trainable_variables())
        with tf.control_dependencies([apply_grad_op]):# ema after updating
            train_ops.append(tf.group(ema_op))
         
    train_op = control_flow_ops.with_dependencies(train_ops, pixel_link_loss, name='train_op')
    return train_op

    
    
def train(train_op):
    summary_op = tf.summary.merge_all()
    sess_config = tf.ConfigProto(log_device_placement = False, allow_soft_placement = True)
    if FLAGS.gpu_memory_fraction < 0:
        sess_config.gpu_options.allow_growth = True
    elif FLAGS.gpu_memory_fraction > 0:
        sess_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction;
    
    init_fn = util.tf.get_init_fn(checkpoint_path = FLAGS.checkpoint_path, train_dir = FLAGS.train_dir, 
                          ignore_missing_vars = FLAGS.ignore_missing_vars, checkpoint_exclude_scopes = FLAGS.checkpoint_exclude_scopes)
    saver = tf.train.Saver(max_to_keep = 500, write_version = 2)
    slim.learning.train(
            train_op,
            logdir = FLAGS.train_dir,
            init_fn = init_fn,
            summary_op = summary_op,
            number_of_steps = FLAGS.max_number_of_steps,
            log_every_n_steps = FLAGS.log_every_n_steps,
            save_summaries_secs = 30,
            saver = saver,
            save_interval_secs = 1200,
            session_config = sess_config
    )


def main(_):
    # The choice of return dataset object via initialization method maybe confusing, 
    # but I need to print all configurations in this method, including dataset information. 
    dataset = config_initialization()   
    
    batch_queue = create_dataset_batch_queue(dataset)
    train_op = create_clones(batch_queue)
    train(train_op)
    
    
if __name__ == '__main__':
    tf.app.run()
