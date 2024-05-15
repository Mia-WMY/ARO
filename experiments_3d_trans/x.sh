#!/bin/bash
#SBATCH --mem=200GB
#SBATCH --partition=ampere
#SBATCH --job-name=part1
#SBATCH --output=job_part4.out
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --error=x.err
#SBATCH --qos=ampere-extd
source activate myenv
python shell_u.py
