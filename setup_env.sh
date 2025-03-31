#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpu
#SBATCH --time=00:20:00
#SBATCH --job-name=create_env_3126
#SBATCH --output=create_env_3126.out

# Load CUDA & conda
uenv verbose cuda-11.4.4 cudnn-11.x-8.8.0
uenv miniconda3-py39

# Set custom, user-writable Conda directories
export CONDA_PKGS_DIRS=$HOME/tmp_conda_pkgs
export CONDA_ENVS_DIRS=$HOME/tmp_conda_envs
mkdir -p $CONDA_PKGS_DIRS $CONDA_ENVS_DIRS

# Avoid touching ~/.conda
export CONDA_ENVS_PATH=$CONDA_ENVS_DIRS
export CONDA_EXE=$(which conda)
unset CONDA_SHLVL
unset CONDA_PREFIX

# Create env using path-based activation
conda create -p $CONDA_ENVS_DIRS/my_env3126 python=3.12.6 -y

# Activate and install dependencies (modify as needed)
source activate $CONDA_ENVS_DIRS/my_env3126

# Example installs — adjust to your actual needs
conda install -c pytorch pytorch torchvision numpy -y
# Or use pip if needed:
# pip install torch torchvision numpy
