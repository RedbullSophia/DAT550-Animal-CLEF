#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpu
#SBATCH --time=20:15:00
#SBATCH --job-name=train3
#SBATCH --output=train3.out

# Set up environment
uenv verbose cuda-11.4.4 cudnn-11.x-8.8.0
uenv miniconda3-py311
python --version

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run your Python script with args
python -u ../model/arg_run.py \
  --backbone resnet18 \
  --batch_size 6144 \
  --num_epochs 10 \
  --lr 0.0001 \
  --m 3 \
  --resize 160 \
  --n 140000
