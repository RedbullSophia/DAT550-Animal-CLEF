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
  --remote \
  --backbone resnet50 \
  --batch_size 64 \
  --num_epochs 30 \
  --lr 0.0005 \
  --m 4 \
  --resize 224 \
  --n 140000 \
  --weight_decay 1e-4 \
  --dropout 0.3 \
  --scheduler cosine \
  --patience 7 \
  --augmentation
