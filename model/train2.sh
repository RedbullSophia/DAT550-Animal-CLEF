#!/bin/bash

# Aggressive regularization with efficientnet_v2_s
python model/arg_run.py \
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