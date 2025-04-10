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
  --batch_size 32 \
  --num_epochs 50 \
  --lr 5e-5 \
  --m 4 \
  --resize 288 \
  --n 140000 \
  --backbone eca_nfnet_l1 \
  --val_split 0.2 \
  --weight_decay 1e-4 \
  --dropout 0.5 \
  --scheduler cosine \
  --patience 15 \
  --augmentation \
  --embedding_dim 512 \
  --margin 0.3 \
  --scale 64.0

