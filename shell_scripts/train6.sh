#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpu
#SBATCH --time=47:15:00
#SBATCH --job-name=train6
#SBATCH --output=train6.out

# Set up environment
uenv verbose cuda-11.4.4 cudnn-11.x-8.8.0
uenv miniconda3-py311
python --version

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


python -u ../model/arg_run.py \
  --remote \
  --batch_size 16 \
  --num_epochs 25 \
  --lr 0.0001 \
  --m 3 \
  --resize 210 \
  --n 140000 \
  --backbone resnet50 \
  --val_split 0.2 \
  --weight_decay 5e-5 \
  --dropout 0.3 \
  --scheduler plateau \
  --patience 10 \
  --augmentation \
  --embedding_dim 512 \
  --margin 0.7 \
  --scale 64.0 \
  --loss_type arcface\
  --reference_model "r50base" \
  --filename r50marg07