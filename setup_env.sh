#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpu
#SBATCH --time=00:15:00
#SBATCH --job-name=create_env_3126
#SBATCH --output=create_env_3126.out

# Load environment modules
uenv verbose cuda-11.4.4 cudnn-11.x-8.8.0
uenv miniconda3-py39  # This gives you access to conda

# Create conda env with Python 3.12.6
conda create -n my_env3126 python=3.12.6 -y

# Activate it
source activate my_env3126

# Install packages
conda install -c pytorch pytorch torchvision -y
# or if packages aren’t yet available for Python 3.12 on conda:
# pip install torch torchvision

# Optional: install from requirements.txt if you have one
# pip install -r requirements.txt
