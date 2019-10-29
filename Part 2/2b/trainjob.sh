#!/bin/sh
#SBATCH -o gpu-job-%j.output
#SBATCH -p RTXq
#SBATCH --gres=gpu:1
#SBATCH -n 1
module load cuda90/toolkit

python3 q2.py
#python3 q3_dropout.py
