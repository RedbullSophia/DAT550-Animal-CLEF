#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpu
#SBATCH --time=47:15:00
#SBATCH --job-name=eval7
#SBATCH --output=eval7.out

# Set up environment
uenv verbose cuda-11.4.4 cudnn-11.x-8.8.0
uenv miniconda3-py311
python --version

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


python -u ../model/evaluate_open_set.py \
  --remote \
  --model_path "/home/stud/aleks99/bhome/DAT550-Animal-CLEF/model_data/2025-04-21_resnet18resize210/trained_model_arcface.pth" \
  --output_dir "/home/stud/aleks99/bhome/DAT550-Animal-CLEF/model_data/2025-04-21_resnet18resize210/open_set_evaluation" \
  --reference_model "2025-04-21_resnet18basemodel" \
  --backbone resnet18 \
  --resize 210 \
  --embedding_dim 512 \
  --batch_size 32 \
  --loss_type arcface