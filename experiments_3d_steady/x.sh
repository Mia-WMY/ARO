#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --job-name=3d_s4000
#SBATCH --output=job_single4000.out
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --error=x.err
#SBATCH --mem=100GB
#SBATCH --nodelist=cn50
#SBATCH --qos=ampere-extd
source activate myenv
python shell.py
