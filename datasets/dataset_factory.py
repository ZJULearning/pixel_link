"""A factory-pattern class which returns classification image/label pairs."""
from datasets import dataset_utils

class DatasetConfig():
    def __init__(self, file_pattern, split_sizes):
        self.file_pattern = file_pattern
        self.split_sizes = split_sizes
        
icdar2013 = DatasetConfig(
        file_pattern = '*_%s.tfrecord', 
        split_sizes = {
            'train': 229,
            'test': 233
        }
)
icdar2015 = DatasetConfig(
        file_pattern = 'icdar2015_%s.tfrecord', 
        split_sizes = {
            'train': 1000,
            'test': 500
        }
)
td500 = DatasetConfig(
        file_pattern = '*_%s.tfrecord', 
        split_sizes = {
            'train': 300,
            'test': 200
        }
)
tr400 = DatasetConfig(
        file_pattern = 'tr400_%s.tfrecord', 
        split_sizes = {
            'train': 400
        }
)
scut = DatasetConfig(
    file_pattern = 'scut_%s.tfrecord',
    split_sizes = {
        'train': 1715
    }
)

synthtext = DatasetConfig(
    file_pattern = '*.tfrecord',
#     file_pattern = 'SynthText_*.tfrecord',
    split_sizes = {
        'train': 858750
    }
)

datasets_map = {
    'icdar2013':icdar2013,
    'icdar2015':icdar2015,
    'scut':scut,
    'td500':td500,
    'tr400':tr400,
    'synthtext':synthtext
}


def get_dataset(dataset_name, split_name, dataset_dir, reader=None):
    """Given a dataset dataset_name and a split_name returns a Dataset.
    Args:
        dataset_name: String, the dataset_name of the dataset.
        split_name: A train/test split dataset_name.
        dataset_dir: The directory where the dataset files are stored.
        reader: The subclass of tf.ReaderBase. If left as `None`, then the default
            reader defined by each dataset is used.
    Returns:
        A `Dataset` class.
    Raises:
        ValueError: If the dataset `dataset_name` is unknown.
    """
    if dataset_name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % dataset_name)
    dataset_config = datasets_map[dataset_name];
    file_pattern = dataset_config.file_pattern
    num_samples = dataset_config.split_sizes[split_name]
    return dataset_utils.get_split(split_name, dataset_dir,file_pattern, num_samples, reader)
