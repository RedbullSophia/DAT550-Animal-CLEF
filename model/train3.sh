#!/bin/bash

# Conservative approach with efficientnet_v2_s
python model/arg_run.py \
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