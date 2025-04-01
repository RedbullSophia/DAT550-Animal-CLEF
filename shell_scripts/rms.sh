#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpu
#SBATCH --time=23:15:00
#SBATCH --job-name=rms
#SBATCH --output=rms.out

# Set up environment
uenv verbose cuda-11.4.4 cudnn-11.x-8.8.0
uenv miniconda3-py311
python --version

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run your Python script with args
python -u ../model/arg_run.py \ 
  --backbone resnet18 \
  --batch_size 4096 \
  --num_epochs 3 \
  --lr 0.0005 \
  --m 2 \
  --resize 128 \
  --n 5000
