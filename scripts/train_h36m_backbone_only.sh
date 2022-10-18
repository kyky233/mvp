#!/bin/bash

#SBATCH -J voxelpose_h36m_train
#SBATCH -p p-V100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH -o /mntnfs/med_data5/wangjiong/3dpose_school/voxelpose-pytorch-copy/slurm_logs/train_backbone_only_%j.out
#SBATCH -e /mntnfs/med_data5/wangjiong/3dpose_school/voxelpose-pytorch-copy/slurm_logs/train_backbone_only_%j.out
#SBATCH --mail-type=ALL  # BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yandq2020@mail.sustech.edu.cn

# export MASTER_PORT=$((12000 + $RANDOM % 2000))
set -x

#CONFIG=configs/h36m/train_h36m_backbone_only_single.yaml
CONFIG=configs/h36m/train_h36m_backbone_only.yaml

# PYTHONPATH="$(dirname ./scripts/train_h36m_backbone_only.sh)/..":$PYTHONPATH \
which python

python -m torch.distributed.launch --nproc_per_node=2 --use_env run/train_2d_backbone.py --cfg $CONFIG
