#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpu
#SBATCH --time=35:15:00
#SBATCH --job-name=train1
#SBATCH --output=train1.out

# Set up environment
uenv verbose cuda-11.4.4 cudnn-11.x-8.8.0
uenv miniconda3-py311
python --version

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


python -u ../model/arg_run.py \
  --remote \
  --batch_size 32 \
  --num_epochs 30 \
  --lr 0.0003 \
  --m 4 \
  --resize 224 \
  --n 140000 \
  --backbone efficientnet_v2_s \
  --val_split 0.2 \
  --weight_decay 1e-4 \
  --dropout 0.4 \
  --scheduler plateau \
  --patience 5 \
  --augmentation \
  --embedding_dim 256 \
  --margin 0.2 \
  --scale 64.0 
