#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpu
#SBATCH --time=35:15:00
#SBATCH --job-name=train2
#SBATCH --output=train2.out

# Set up environment
uenv verbose cuda-11.4.4 cudnn-11.x-8.8.0
uenv miniconda3-py311
python --version

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

 
python -u ../model/arg_run.py \
  --remote \
  --batch_size 64 \
  --num_epochs 30 \
  --lr 0.001 \
  --m 8 \
  --resize 224 \
  --n 140000 \
  --backbone efficientnet_v2_s \
  --val_split 0.2 \
  --weight_decay 1e-3 \
  --dropout 0.6 \
  --scheduler plateau \
  --patience 5 \
  --augmentation \
  --embedding_dim 512 \
  --margin 0.3 \
  --scale 64.0 
