#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpu
#SBATCH --time=23:15:00
#SBATCH --job-name=dwl
#SBATCH --output=dwl.out

# Set up environment
uenv verbose cuda-11.4.4 cudnn-11.x-8.8.0
uenv miniconda3-py311
python --version
#pip install -r requirements4.txt
#pip install --force-reinstall numpy==2.2.4 scikit-learn==1.6.1 pandas==2.2.3 bottleneck
#pip install bottleneck --upgrade

# Run your Python script with args
python download.py 
