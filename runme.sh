#!/bin/bash
DATASET_DIR="/vol/vssp/datasets/audio/dcase2018/task4/dataset"
WORKSPACE="/vol/vssp/msos/qk/workspaces/pub_dcase2018_task4"

BACKEND="pytorch"
GPU_ID=0

# Calculate features
python utils/features.py logmel --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --data_type='dev_train_weak'
python utils/features.py logmel --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --data_type='dev_train_unlabel_out_of_domain'
python utils/features.py logmel --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --data_type='dev_train_unlabel_in_domain'
python utils/features.py logmel --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --data_type='dev_test'
python utils/features.py logmel --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --data_type='eval'

############ Development ############
# Train
CUDA_VISIBLE_DEVICES=$GPU_ID python $BACKEND/main_pytorch.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --validate --cuda

# Inference
CUDA_VISIBLE_DEVICES=$GPU_ID python $BACKEND/main_pytorch.py inference_validation_data --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --iteration=3000 --cuda

############ Full train ############
# Train
CUDA_VISIBLE_DEVICES=$GPU_ID python $BACKEND/main_pytorch.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --cuda

# Inference
CUDA_VISIBLE_DEVICES=$GPU_ID python $BACKEND/main_pytorch.py inference_testing_data --workspace=$WORKSPACE --iteration=3000 --cuda