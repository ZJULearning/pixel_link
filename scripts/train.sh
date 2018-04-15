set -x
set -e
export CUDA_VISIBLE_DEVICES=$1
IMG_PER_GPU=$2

TRAIN_DIR=${HOME}/models/pixel_link

# get the number of gpus
OLD_IFS="$IFS" 
IFS="," 
gpus=($CUDA_VISIBLE_DEVICES) 
IFS="$OLD_IFS"
NUM_GPUS=${#gpus[@]}

# batch_size = num_gpus * IMG_PER_GPU
BATCH_SIZE=`expr $NUM_GPUS \* $IMG_PER_GPU`

#DATASET=synthtext
#DATASET_PATH=SynthText

DATASET=icdar2015
DATASET_DIR=$HOME/dataset/pixel_link/icdar2015

python train_pixel_link.py \
            --train_dir=${TRAIN_DIR} \
            --num_gpus=${NUM_GPUS} \
            --learning_rate=1e-3\
            --gpu_memory_fraction=-1 \
            --train_image_width=512 \
            --train_image_height=512 \
            --batch_size=${BATCH_SIZE}\
            --dataset_dir=${DATASET_DIR} \
            --dataset_name=${DATASET} \
            --dataset_split_name=train \
            --max_number_of_steps=100\
            --checkpoint_path=${CKPT_PATH} \
            --using_moving_average=1

python train_pixel_link.py \
            --train_dir=${TRAIN_DIR} \
            --num_gpus=${NUM_GPUS} \
            --learning_rate=1e-2\
            --gpu_memory_fraction=-1 \
            --train_image_width=512 \
            --train_image_height=512 \
            --batch_size=${BATCH_SIZE}\
            --dataset_dir=${DATASET_DIR} \
            --dataset_name=${DATASET} \
            --dataset_split_name=train \
            --checkpoint_path=${CKPT_PATH} \
            --using_moving_average=1\
            2>&1 | tee -a ${TRAIN_DIR}/log.log                        

