#!/bin/bash

# Set up environment
python --version

# Set CUDA memory allocation config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the training script with local parameters
cd ../model && python -u train.py \
  --backbone resnet50 \
  --batch_size 64 \
  --num_epochs 1 \
  --lr 0.0005 \
  --m 4 \
  --resize 160 \
  --n 5000 \
  --val_split 0.2 \
  --weight_decay 5e-4 \
  --dropout 0.5 \
  --scheduler cosine \
  --patience 10 \
  --augmentation \
  --embedding_dim 512 \
  --margin 0.4 \
  --scale 64.0 