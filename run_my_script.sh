#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --job-name=run_my_script
#SBATCH --output=run_my_script.out

# Load environment
uenv verbose cuda-11.4.4 cudnn-11.x-8.8.0
uenv miniconda3-py39
source activate my_env3126

# Run your script
python -u model/arg_run.py \
    --batch_size 1024 \
    --num_epochs 1 \
    --lr 0.0005 \
    --m 2 \
    --resize 128 \
    --n 4000