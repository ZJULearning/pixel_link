"""
Taken a fscore_csv file, create a tfsummary event file which can used with tensorboard
to visualize the fscore curve.

The tf summary event file will be generated in the same folder as the fscore_csv file

"""
import argparse
import os

import pandas as pd
import tensorflow as tf
tf.enable_eager_execution()


def main(args):
    output_dir = os.path.dirname(args.fscore_csv_fpath)
    summary_writer = tf.contrib.summary.create_file_writer(output_dir)

    df = pd.read_csv(args.fscore_csv_fpath, header=None, names=['step', 'recall', 'precision', 'f1_score'])
    with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
        for row in df.itertuples():
            tf.contrib.summary.scalar("recall", row.recall, step=row.step)
            tf.contrib.summary.scalar("precision", row.precision, step=row.step)
            tf.contrib.summary.scalar("f1_score", row.f1_score, step=row.step)
        tf.contrib.summary.flush()
        print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--fscore_csv_fpath", required=True)
    args = parser.parse_args()
    main(args)