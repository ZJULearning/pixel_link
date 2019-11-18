"""
This script can be used in two mode
1) one-off mode
You specify the exact checkpoint to evaluate, the script will print the result and quit.

2) watching mode
You specify the checkpoint base dir, the script will poll the folder for new checkpoint. When
a new checkpoint is generated, the script will evaluate performance with this checkpoint and
 append the result into fscore.csv in the checkpoint base dir
"""
import argparse
import glob
import re
import subprocess
import time
import os

# this script itself only use tensorflow for creating tf.summary
# so no GPU is required. But we will require a GPU,
# which is to be specified in the cuda_visible_devices command option,  to do the inference later.

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
tf.enable_eager_execution()

EVALUATION_INTERVAL_IN_MINUTES = 20


def my_exec(cmd):
    print(cmd)
    os.system(cmd)


def exec_and_get_stdout(eval_fscore_cmd):
    print(eval_fscore_cmd)
    res = subprocess.check_output(eval_fscore_cmd, stderr=subprocess.STDOUT, shell=True, universal_newlines=True)
    print(res)
    return res


def extract_value(stdout_txt, metric_name):
    pattern = r'.*"%s": ((0\.)?\d*),.*' % metric_name
    return float(re.match(pattern, stdout_txt).group(1))


def eval_with_checkpoint(checkpoint_base_name, args):
    """

    :param checkpoint_base_name:
    :param global_step:
    :param args:
    :return: (recall, precision, fscore)
    """
    checkpoint_path = os.path.join(args.checkpoint_folder, checkpoint_base_name)

    inference_cmd = '''CUDA_VISIBLE_DEVICES=%s PYTHONPATH=$PYTHONPATH:./pylib/src python test_pixel_link.py \
     --checkpoint_path=%s \
     --dataset_dir=%s \
     --gpu_memory_fraction=-1    
''' % (args.cuda_visible_devices, checkpoint_path, args.test_image_folder)

    my_exec(inference_cmd)
    inference_res_zip_file = os.path.join(
        args.checkpoint_folder, 'test', checkpoint_base_name,
        '%s_det.zip' % checkpoint_base_name
    )

    eval_fscore_cmd = '''
    PYTHON_PATH=$PYTHONPATH:evaluation_script python ./evaluation_script/script.py -g=%s -s=%s
    ''' % (args.test_ground_truth_zip_file, inference_res_zip_file)

    stdout_txt = exec_and_get_stdout(eval_fscore_cmd)

    recall = extract_value(stdout_txt, 'recall')
    precision = extract_value(stdout_txt, 'precision')
    fscore = extract_value(stdout_txt, 'hmean')
    return recall, precision, fscore


def get_latest_checkpoint(checkpoint_folder):
    """

    :param checkpoint_folder:
    :return: None  if there is no any checkpoint in the folder yet,
    otherwise a tuple of  latest_checkpoint_base_name, latest_global_step


    """
    step_number_regex = re.compile(r'.*model\.ckpt-(\d*)\.index')
    step_numbers = [int(step_number_regex.match(f).group(1)) for f in glob.glob('%s/model.ckpt-*.index' % checkpoint_folder)]

    if len(step_numbers) == 0:
        return None

    latest_global_step = sorted(step_numbers)[-1]
    latest_checkpoint_base_name = 'model.ckpt-%s' % latest_global_step
    return latest_checkpoint_base_name, latest_global_step


def main(args):
    if args.checkpoint_path is not None:
        one_off_evaluate(args)
    else:
        watching_and_evaluate(args)


def watching_and_evaluate(args):
    tf_summary_writer = tf.contrib.summary.create_file_writer(args.checkpoint_folder)

    last_evaluated_global_step = None
    while True:
        res = get_latest_checkpoint(args.checkpoint_folder)
        if res is not None:
            latest_checkpoint_base_name, latest_global_step = res
            # avoid evaluating on the very first checkpoint
            if latest_global_step > 0 and last_evaluated_global_step != latest_global_step:
                print('evaluate with %s' % latest_checkpoint_base_name)
                recall, precision, fscore = eval_with_checkpoint(latest_checkpoint_base_name, args)

                # append result to fscrore.csv
                log_result_to_file(recall, precision, fscore, latest_global_step, args)
                log_result_for_tensorboard(recall, precision, fscore, latest_global_step, tf_summary_writer)

                last_evaluated_global_step = latest_global_step

        print('sleep ...')
        time.sleep(EVALUATION_INTERVAL_IN_MINUTES * 60)


def one_off_evaluate(args):
    args.checkpoint_folder, checkpoint_base_name = args.checkpoint_path.rsplit('/', 1)
    recall, precision, fscore = eval_with_checkpoint(checkpoint_base_name, args)
    print('recall=%.6f, prections=%.6f, fscore=.6f' % (recall, precision, fscore))


def log_result_to_file(recall, precision, fscore, latest_global_step, args):
    fscore_csv_fpath = os.path.join(args.checkpoint_folder, 'fscore.csv')
    row = '%d,%f,%f,%f' % (latest_global_step, recall, precision, fscore)
    file(fscore_csv_fpath, 'a').write(row + '\n')


def log_result_for_tensorboard(recall, precision, fscore, step, tf_summary_writer):
    with tf_summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar("val/recall", recall, step=step)
        tf.contrib.summary.scalar("val/precision", precision, step=step)
        tf.contrib.summary.scalar("val/fscore", fscore, step=step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        "--cuda_visible_devices",
        default='2',
        help='Specify a gpu id which is not used by the training job',
    )
    parser.add_argument(
        "-c",
        "--checkpoint_folder",
        default='./checkpoint',
        help='if this option is specified, the script will run in watching mode'
    )
    parser.add_argument(
        "-p",
        "--checkpoint_path",
        help='if this option is specified, the script will run in one-off mode '
    )
    parser.add_argument(
        "-t", "--test_image_folder",
        default='/home/victor/workspace/datasets/scene_text/street_number_recognition/val/jpg.clear',
    )
    parser.add_argument(
        "-z", "--test_ground_truth_zip_file",
        default='/home/victor/workspace/datasets/scene_text/street_number_recognition/val/ic15_format_label.clear.zip',
    )
    args = parser.parse_args()
    main(args)