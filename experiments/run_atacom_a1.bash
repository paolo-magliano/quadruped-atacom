#!/bin/bash
#SBATCH -p stud
#SBATCH -t 24:00:00
#SBATCH -c 3
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH -o outputs/%j/logs.txt
#SBATCH -e outputs/%j/errors.txt
#SBATCH --mail-type=END

source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaacsim
python experiments/run_exp_a1.py env_type=PD constraints.joint_limit=[1.]
python experiments/run_exp_a1.py env_type=PD constraints.joint_limit=[2.]
