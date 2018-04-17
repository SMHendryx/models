#!/usr/bin/env bash

# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Run quick test:
# From tensorflow/models/research/
#python deeplab/model_test.py

# A local visualization job using xception_65 can be run with the following command:

# vars:
#PATH_TO_CHECKPOINT=/home/sean/repositories/tensorflow_models/models/research/deeplab/checkpoints/deeplabv3_cityscapes_train/
PATH_TO_CHECKPOINT=/home/sean/repositories/tensorflow_models/models/research/deeplab/checkpoints/deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz
PATH_TO_VIS_DIR=/home/sean/data/experiments/deeplab/xception_cityscapes_trainfine/apolloscape/road02_ins
PATH_TO_DATASET=/home/sean/repositories/tensorflow_models/models/research/deeplab/datasets/apolloscape/one_image

# Check if direc exists
if [[ ! -f $PATH_TO_CHECKPOINT ]]; then
    echo "ERROR: $PATH_TO_CHECKPOINT is not a directory" 1>&2
    exit 1
fi

# Check if direc exists
if [[ ! -d $PATH_TO_VIS_DIR ]]; then
    echo "ERROR: $PATH_TO_VIS_DIR is not a directory" 1>&2
    exit 1
fi

# Check if direc exists
if [[ ! -d $PATH_TO_DATASET ]]; then
    echo "ERROR: $PATH_TO_DATASET is not a directory" 1>&2
    exit 1
fi

# From tensorflow/models/research/
python deeplab/vis.py \
    --logtostderr \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --vis_crop_size=1025 \
    --vis_crop_size=2049 \
    --dataset="cityscapes" \
    --colormap_type="cityscapes" \
    --checkpoint_dir=${PATH_TO_CHECKPOINT} \
    --vis_logdir=${PATH_TO_VIS_DIR} \
    --dataset_dir=${PATH_TO_DATASET} \
    --also_save_raw_predictions = True \
    --vis_split="val" \

# Where ${PATH_TO_CHECKPOINT} is the path to the trained checkpoint (i.e., the path to train_logdir), ${PATH_TO_VIS_DIR} is the directory in which evaluation events will be written to, and ${PATH_TO_DATASET} is the directory in which the Cityscapes dataset resides. Note that if the users would like to save the segmentation results for evaluation server, set also_save_raw_predictions = True.
