#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpu
#SBATCH --time=20:15:00
#SBATCH --job-name=train2
#SBATCH --output=train2.out

# Set up environment
uenv verbose cuda-11.4.4 cudnn-11.x-8.8.0
uenv miniconda3-py311
python --version

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run your Python script with args - ResNet18 with different approach
python -u ../model/arg_run.py \
  --remote \
  --backbone resnet18 \
  --batch_size 64 \
  --num_epochs 30 \
  --lr 0.0005 \
  --m 8 \
  --resize 256 \
  --n 140000 \
  --weight_decay 1e-4 \
  --dropout 0.3 \
  --scheduler plateau \
  --patience 7 \
  --augmentation \
  --embedding_dim 512 \
  --margin 0.4 \
  --scale 32.0
