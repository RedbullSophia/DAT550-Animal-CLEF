#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpu
#SBATCH --time=47:15:00
#SBATCH --job-name=test
#SBATCH --output=test.out

# Set up environment
uenv verbose cuda-11.4.4 cudnn-11.x-8.8.0
uenv miniconda3-py311
python --version

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


python -u ../model/arg_run.py \
  --remote \
  --batch_size 32 \
  --num_epochs 1 \
  --lr 0.0001 \
  --m 4 \
  --resize 288 \
  --n 2000 \
  --backbone resnet18 \
  --val_split 0.2 \
  --weight_decay 5e-5 \
  --dropout 0.3 \
  --scheduler cosine \
  --patience 10 \
  --augmentation \
  --embedding_dim 512 \
  --margin 0.3 \
  --scale 64.0 \
  --loss_type arcface\
  --filename resnet18testmodel
