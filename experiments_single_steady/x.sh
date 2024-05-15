#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --job-name=hello2
#SBATCH --output=job_single.out
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --error=x.err
#SBATCH --qos=ampere-extd
source activate myenv
python union_test.py
