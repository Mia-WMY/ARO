#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --job-name=new_score8
#SBATCH --output=newjob.out
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --error=x.err
#SBATCH --nodelist=cn50
#SBATCH --qos=ampere-extd
source activate myenv
python shell.py
