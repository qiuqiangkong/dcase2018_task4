#!/bin/bash
DATASET_DIR="/vol/vssp/datasets/audio/dcase2018/task4/dataset"
WORKSPACE="/vol/vssp/msos/qk/workspaces/pub_dcase2018_task4"

# Calculate features
python features.py logmel --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --data_type='dev_train_weak'
python features.py logmel --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --data_type='dev_train_unlabel_out_of_domain'
python features.py logmel --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --data_type='dev_train_unlabel_in_domain'
python features.py logmel --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --data_type='dev_test'
python features.py logmel --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --data_type='eval'

BACKEND="pytorch"

############ Development ############
# Train
CUDA_VISIBLE_DEVICES=0 python $BACKEND/main_pytorch.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --validate --cuda

# Inference
CUDA_VISIBLE_DEVICES=0 python $BACKEND/main_pytorch.py inference_validation --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --iteration=5000 --cuda

############ Full train ############
# Train
CUDA_VISIBLE_DEVICES=0 python $BACKEND/main_pytorch.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --cuda

# Inference
CUDA_VISIBLE_DEVICES=0 python $BACKEND/main_pytorch.py inference_testing_data --workspace=$WORKSPACE --iteration=5000 --cuda