#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpu
#SBATCH --time=02:15:00
#SBATCH --job-name=runscript
#SBATCH --output=ran.out

# Set up environment
uenv verbose cuda-11.4.4 cudnn-11.x-8.8.0
uenv miniconda3-py311
python --version
#pip install -r requirements4.txt
#pip install --force-reinstall numpy==2.2.4 scikit-learn==1.6.1 pandas==2.2.3 bottleneck
#pip install bottleneck --upgrade

# Run your Python script with args
python -u model/arg_run.py \
  --batch_size 8192 \
  --num_epochs 10 \
  --lr 0.0005 \
  --m 4 \
  --resize 210 \
  --n 40000