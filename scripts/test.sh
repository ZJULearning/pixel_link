set -x
set -e
export CUDA_VISIBLE_DEVICES=$1
python test_pixel_link.py \
     --checkpoint_path=$2 \
     --dataset_dir=$3\
     --gpu_memory_fraction=-1
     

