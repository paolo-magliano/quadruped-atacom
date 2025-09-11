#!/bin/bash
#SBATCH -p stud
#SBATCH -t 24:00:00
#SBATCH -c 1
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH -o outputs/%j/logs.txt
#SBATCH -e outputs/%j/errors.txt

source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaacsim

python experiments/plot/plot_metric.py --cfg_paths /home/stud_magliano/projects/SafeLocomotion/logs/Foot_rot_0.3_1.57_2_NI