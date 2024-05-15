#!/bin/bash
#SBATCH --mem=100GB
#SBATCH --partition=ampere
#SBATCH --job-name=400_100
#SBATCH --output=400_100.out
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --error=x.err
#SBATCH --qos=ampere-extd
source activate myenv
python shell.py
