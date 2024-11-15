#!/bin/bash
#SBATCH --job-name=example
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=1000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source load.sh

srun bash notebook.sh
