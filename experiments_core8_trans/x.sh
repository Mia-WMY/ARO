#!/bin/bash
#SBATCH --mem=100GB
#SBATCH --partition=ampere
#SBATCH --job-name=core8_trans_union
#SBATCH --output=job_test.out
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --error=x.err
#SBATCH --qos=ampere-extd
source activate myenv
python union_test.py
