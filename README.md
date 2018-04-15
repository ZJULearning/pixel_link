Code for the AAAI18 paper [PixelLink: Detecting Scene Text via Instance Segmentation](https://arxiv.org/abs/1801.01315).
# Installation
## Clone the repo to your computer:
```
git clone --recursive git@github.com:ZJULearning/pixel_link.git
```

Denote the root directory path of pixel_link by ${pixel_link_root}. 

Add the path of `${pixel_link_root}/pylib/src` to your `PYTHONPATH`:
```
export PYTHONPATH=${path_to_pixel_link}/pylib/src:$PYTHONPATH
```

## Prerequisites
 Tested on Ubuntu14.04 and 16.04 with:
* Python 2.7
* Tensorflow-gpu >= 1.1
* opencv2
* setproctitle
* matplotlib

Anaconda is recommended to for an easier installation:

1. Install [Anaconda](https://anaconda.org/)
2. Create and activate the required virtual environment by:
```
conda env create --file pixel_link_env.txt
source activate pixel_link
```

# Testing
## Download the pretrained model
* PixelLink + VGG16 4s, trained on IC15:[Baidu Net Disk](https://pan.baidu.com/s/1jsOc-cutC4GyF-wMMyj5-w)
* PixelLink + VGG16 2s, trained on IC15:[Baidu Net Disk](https://pan.baidu.com/s/1asSFsRSgviU2GnvGt2lAUw)

Unzip the downloaded model. It contains 4 files:

* config.py
* model.ckpt-xxx.data-00000-of-00001
* model.ckpt-xxx.index  
* model.ckpt-xxx.meta

Denote their parent directory as ${model_path}.

## Test on ICDAR2015
Suppose you have downloaded the [ICDAR2015 dataset](http://rrc.cvc.uab.es/?ch=4&com=downloads), execute the following commands to test the model on ICDAR2015:
```
cd ${pixel_link_root}
./scripts/test.sh ${GPU_ID} ${model_path}/model.ckpt-xxx ${path_to_icdar2015}/ch4_test_images
```
For example:
```
./scripts/test.sh 3 ~/temp/conv3_3/model.ckpt-38055 ~/dataset/ICDAR2015/Challenge4/ch4_test_images
```

The program will create a zip file of  detection results, which can be submitted to the ICDAR2015 server directly.
The detection results can be visualized via `scripts/vis.sh`.

## Test on any images
Put the images to be tested in a single directory, i.e., ${image_dir}. Then:
```
cd ${pixel_link_root}
./scripts/test_any.sh ${GPU_ID} ${model_path}/model.ckpt-xxx ${image_dir}
```
For example:
```
 ./scripts/test_any.sh 3 ~/temp/conv3_3/model.ckpt-38055 ~/dataset/ICDAR2015/Challenge4/ch4_training_images
```

The program will visualize the detection results directly on images.   If the detection result is not satisfying:

1. Adjust the inference parameters like `eval_image_width`, `eval_image_height`, `pixel_conf_threshold`, `link_conf_threshold`.
2. Or train your own model.

# Training
## Converting the dataset to tfrecords files
Scripts for converting ICDAR2015 and SynthText dataset have been provided in the `datasets` directory.
 It not hard to write a converting script  for your own dataset.

## Train your own model

* Modify `scripts/train.sh` to configure your dataset name and dataset path like:
```
DATASET=icdar2015
DATASET_DIR=$HOME/dataset/pixel_link/icdar2015
```
* Start training
```
./scripts/train.sh ${GPU_IDs} ${IMG_PER_GPU}
```
For example, `./scripts/train.sh 0,1,2 8`. 

The existing training strategy is configured for icdar2015, modify it if necessary.

# Acknowlegement
![](http://www.cad.zju.edu.cn/templets/default/imgzd/logo.jpg)
![](http://www.cvte.com/images/logo.png)
