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

# python experiments/run_exp_a1.py constraints.foot_pos_max_z=-0.2 test=True render=True record=True checkpoint=/home/stud_magliano/projects/SafeLocomotion/logs/A1_2025-04-06-20-16-54
# python experiments/run_exp_a1.py constraints.foot_pos_max_z=-0.3 test=True render=True record=True checkpoint=/home/stud_magliano/projects/SafeLocomotion/logs/A1_2025-04-06-20-16-54
# python experiments/run_exp_a1.py constraints.foot_pos_min_z=-0.2 test=True render=True record=True checkpoint=/home/stud_magliano/projects/SafeLocomotion/logs/A1_2025-04-06-20-16-54
# python experiments/run_exp_a1.py constraints.foot_pos_min_z=-0.3 test=True render=True record=True checkpoint=/home/stud_magliano/projects/SafeLocomotion/logs/A1_2025-04-06-20-16-54

# python experiments/run_exp_a1.py constraints.foot_pos_max_z=-0.2 constraints.foot_pos_min_z=-0.4 constraints.foot_pos_alpha=''
# python experiments/run_exp_a1.py constraints.foot_pos_max_z=-0.3 constraints.foot_pos_min_z=-0.6 constraints.foot_pos_alpha=''
# python experiments/run_exp_a1.py constraints.foot_pos_max_z=0.3 constraints.foot_pos_min_z=-0.25 constraints.foot_pos_alpha=''

# python experiments/run_exp_a1.py atacom.enable=False

# python experiments/run_exp_a1.py constraints.foot_pos_max_z=-0.3 checkpoint=/home/stud_magliano/projects/SafeLocomotion/logs/A1_2025-04-06-20-16-54

# python experiments/run_exp_a1.py constraints.foot_pos_max_z=-0.15 constraints.foot_pos_min_z='' constraints.foot_pos_alpha='' train.params.ent_coeff=1e-3 train.atacom_policy=False
# python experiments/run_exp_a1.py constraints.foot_pos_max_z=-0.15 constraints.foot_pos_min_z='' constraints.foot_pos_alpha='' train.params.ent_coeff=1e-2 train.atacom_policy=False
# python experiments/run_exp_a1.py constraints.foot_pos_max_z=-0.15 constraints.foot_pos_min_z='' constraints.foot_pos_alpha='' train.params.ent_coeff=1e-1 train.atacom_policy=False

# python experiments/run_exp_a1.py constraints.foot_pos_max_z=0. constraints.foot_pos_min_z='' constraints.foot_pos_alpha='' train.params.ent_coeff=1e-2 atacom.slack_beta=10
# python experiments/run_exp_a1.py constraints.foot_pos_max_z=0.1 constraints.foot_pos_min_z='' constraints.foot_pos_alpha='' train.params.ent_coeff=5e-4 atacom.slack_beta=10 train.atacom_policy=False

# python experiments/run_exp_a1.py constraints.foot_pos_max_z=0. constraints.foot_pos_min_z='' constraints.foot_pos_alpha='' train.params.ent_coeff=1e-3 atacom.slack_beta=10
# python experiments/run_exp_a1.py constraints.foot_pos_max_z=-0.15 constraints.foot_pos_min_z='' constraints.foot_pos_alpha='' train.params.ent_coeff=1e-3 atacom.slack_beta=10 train.atacom_policy=False
# python experiments/run_exp_a1.py constraints.foot_pos_max_z=-0.15 constraints.foot_pos_min_z='' constraints.foot_pos_alpha='' train.params.ent_coeff=1e-3 atacom.slack_beta=10 train.atacom_policy=False

# python experiments/run_exp_a1.py constraints.foot_pos_max_z='' constraints.foot_pos_min_z=-0.4 constraints.foot_pos_alpha='' train.params.ent_coeff=1e-4 atacom.slack_beta=10 train.atacom_policy=False
# python experiments/run_exp_a1.py constraints.foot_pos_max_z='' constraints.foot_pos_min_z=-0.5 constraints.foot_pos_alpha='' train.params.ent_coeff=1e-4 atacom.slack_beta=10 train.atacom_policy=False
# python experiments/run_exp_a1.py constraints.foot_pos_max_z='' constraints.foot_pos_min_z=-0.6 constraints.foot_pos_alpha='' train.params.ent_coeff=1e-4 atacom.slack_beta=10 train.atacom_policy=False

# python experiments/run_exp_a1.py constraints.foot_rot_base='[[0., 0., 0.]]' constraints.foot_rot_min=1.57 constraints.foot_rot_max=3.14 constraints.foot_pos_max_z='' constraints.foot_pos_min_z='' constraints.foot_pos_alpha='' train.params.ent_coeff=1e-4 atacom.slack_beta=10 train.atacom_policy=False
# python experiments/run_exp_a1.py constraints.foot_rot_base='[[0., 0., 0.]]' constraints.foot_rot_min=1.57 constraints.foot_rot_max=3.14 constraints.foot_pos_max_z='' constraints.foot_pos_min_z='' constraints.foot_pos_alpha='' train.params.ent_coeff=1e-5 atacom.slack_beta=10 train.atacom_policy=False
# python experiments/run_exp_a1.py constraints.foot_rot_base='[[0., -0.3, 0.]]' constraints.foot_rot_min=1.57 constraints.foot_rot_max=3.14 constraints.foot_pos_max_z='' constraints.foot_pos_min_z='' constraints.foot_pos_alpha='' train.params.ent_coeff=1e-4 atacom.slack_beta=10 train.atacom_policy=False
# python experiments/run_exp_a1.py constraints.foot_rot_base='[[0., -0.3, 0.]]' constraints.foot_rot_min=1.57 constraints.foot_rot_max=3.14 constraints.foot_pos_max_z='' constraints.foot_pos_min_z='' constraints.foot_pos_alpha='' train.params.ent_coeff=1e-5 atacom.slack_beta=10 train.atacom_policy=False

# python experiments/run_exp_a1.py constraints.foot_pos_max_z='' constraints.foot_pos_min_z=-0.4 constraints.foot_pos_alpha='' train.params.ent_coeff=5e-5 atacom.slack_beta=10 train.atacom_policy=False
# python experiments/run_exp_a1.py constraints.foot_pos_max_z='' constraints.foot_pos_min_z=-0.4 constraints.foot_pos_alpha='' train.params.ent_coeff=1e-5 atacom.slack_beta=10 train.atacom_policy=False
# python experiments/run_exp_a1.py constraints.foot_pos_max_z='' constraints.foot_pos_min_z=-0.3 constraints.foot_pos_alpha='' train.params.ent_coeff=1e-4 atacom.slack_beta=10 train.atacom_policy=False
# python experiments/run_exp_a1.py constraints.foot_pos_max_z='' constraints.foot_pos_min_z=-0.3 constraints.foot_pos_alpha='' train.params.ent_coeff=5e-5 atacom.slack_beta=10 train.atacom_policy=False
# python experiments/run_exp_a1.py constraints.foot_pos_max_z='' constraints.foot_pos_min_z=-0.3 constraints.foot_pos_alpha='' train.params.ent_coeff=1e-5 atacom.slack_beta=10 train.atacom_policy=False

# python experiments/run_exp_a1.py constraints.foot_pos_max_z='' constraints.foot_pos_min_z=-0.2 constraints.foot_pos_alpha='' train.params.ent_coeff=1e-4 atacom.slack_beta=10 train.atacom_policy=False
# python experiments/run_exp_a1.py constraints.foot_pos_max_z='' constraints.foot_pos_min_z=-0.2 constraints.foot_pos_alpha='' train.params.ent_coeff=5e-5 atacom.slack_beta=10 train.atacom_policy=False
# python experiments/run_exp_a1.py constraints.foot_pos_max_z='' constraints.foot_pos_min_z=-0.2 constraints.foot_pos_alpha='' train.params.ent_coeff=1e-5 atacom.slack_beta=10 train.atacom_policy=False

# python experiments/run_exp_a1.py constraints.foot_pos_max_z='' constraints.foot_pos_min_z=-0.1 constraints.foot_pos_alpha='' train.params.ent_coeff=1e-4 atacom.slack_beta=10 train.atacom_policy=False
# python experiments/run_exp_a1.py constraints.foot_pos_max_z='' constraints.foot_pos_min_z=-0.1 constraints.foot_pos_alpha='' train.params.ent_coeff=5e-5 atacom.slack_beta=10 train.atacom_policy=False
# python experiments/run_exp_a1.py constraints.foot_pos_max_z='' constraints.foot_pos_min_z=-0.1 constraints.foot_pos_alpha='' train.params.ent_coeff=1e-5 atacom.slack_beta=10 train.atacom_policy=False

# python experiments/run_exp_a1.py constraints.foot_rot_base='[[0., -0.4, 0.]]' constraints.foot_rot_min=1.57 constraints.foot_rot_max=3.14 constraints.foot_pos_max_z='' constraints.foot_pos_min_z='' constraints.foot_pos_alpha='' train.params.ent_coeff=1e-4 atacom.slack_beta=10 train.atacom_policy=False
# python experiments/run_exp_a1.py constraints.foot_rot_base='[[0., -0.5, 0.]]' constraints.foot_rot_min=1.57 constraints.foot_rot_max=3.14 constraints.foot_pos_max_z='' constraints.foot_pos_min_z='' constraints.foot_pos_alpha='' train.params.ent_coeff=1e-4 atacom.slack_beta=10 train.atacom_policy=False
# python experiments/run_exp_a1.py constraints.foot_rot_base='[[0., -0.6, 0.]]' constraints.foot_rot_min=1.57 constraints.foot_rot_max=3.14 constraints.foot_pos_max_z='' constraints.foot_pos_min_z='' constraints.foot_pos_alpha='' train.params.ent_coeff=1e-4 atacom.slack_beta=10 train.atacom_policy=False
# python experiments/run_exp_a1.py constraints.foot_rot_base='[[0., -0.3, 0.]]' constraints.foot_rot_min=1.4 constraints.foot_rot_max=3.14 constraints.foot_pos_max_z='' constraints.foot_pos_min_z='' constraints.foot_pos_alpha='' train.params.ent_coeff=1e-4 atacom.slack_beta=10 train.atacom_policy=False
# python experiments/run_exp_a1.py constraints.foot_rot_base='[[0., -0.3, 0.]]' constraints.foot_rot_min=1.57 constraints.foot_rot_max=3.14 constraints.foot_pos_max_z=-0.3 constraints.foot_pos_min_z='' constraints.foot_pos_alpha='' train.params.ent_coeff=1e-3 atacom.slack_beta=10 train.atacom_policy=False

# python experiments/run_exp_a1.py constraints.foot_pos_max_z=-0.2 constraints.foot_pos_min_z='' constraints.foot_pos_alpha='' train.params.ent_coeff=1e-3 atacom.slack_beta=10 train.atacom_policy=False
# python experiments/run_exp_a1.py constraints.foot_pos_max_z=-0.2 constraints.foot_pos_min_z='' constraints.foot_pos_alpha='' train.params.ent_coeff=1e-4 atacom.slack_beta=10 train.atacom_policy=False
# python experiments/run_exp_a1.py constraints.foot_pos_max_z=-0.35 constraints.foot_pos_min_z='' constraints.foot_pos_alpha='' train.params.ent_coeff=1e-3 atacom.slack_beta=10 train.atacom_policy=False
# python experiments/run_exp_a1.py constraints.foot_pos_max_z=-0.35 constraints.foot_pos_min_z='' constraints.foot_pos_alpha='' train.params.ent_coeff=1e-4 atacom.slack_beta=10 train.atacom_policy=False

# python experiments/run_exp_a1.py atacom.enable=False control.type='Vel' control.action_scale=0.75 train.params.ent_coeff=0.001 wandb.project='ATACOM_A1_Vel'
# python experiments/run_exp_a1.py atacom.enable=False control.type='Vel' control.action_scale=1. train.params.ent_coeff=0.001 wandb.project='ATACOM_A1_Vel'
# python experiments/run_exp_a1.py atacom.enable=False control.type='Vel' control.action_scale=1.25 train.params.ent_coeff=0.001 wandb.project='ATACOM_A1_Vel'
# python experiments/run_exp_a1.py atacom.enable=False control.type='Vel' control.action_scale=1.5 train.params.ent_coeff=0.001 wandb.project='ATACOM_A1_Vel'
# python experiments/run_exp_a1.py atacom.enable=False control.type='Vel' control.action_scale=2. train.params.ent_coeff=0.001 wandb.project='ATACOM_A1_Vel'
# python experiments/run_exp_a1.py atacom.enable=False control.type='Vel' control.action_scale=2.5 train.params.ent_coeff=0.001 wandb.project='ATACOM_A1_Vel'
# python experiments/run_exp_a1.py atacom.enable=False control.type='Vel' control.action_scale=0.75 train.params.ent_coeff=0.0001 wandb.project='ATACOM_A1_Vel'
# python experiments/run_exp_a1.py atacom.enable=False control.type='Vel' control.action_scale=1. train.params.ent_coeff=0.0001 wandb.project='ATACOM_A1_Vel'
# python experiments/run_exp_a1.py atacom.enable=False control.type='Vel' control.action_scale=1.25 train.params.ent_coeff=0.0001 wandb.project='ATACOM_A1_Vel'
# python experiments/run_exp_a1.py atacom.enable=False control.type='Vel' control.action_scale=1.5 train.params.ent_coeff=0.0001 wandb.project='ATACOM_A1_Vel'
# python experiments/run_exp_a1.py atacom.enable=False control.type='Vel' control.action_scale=2. train.params.ent_coeff=0.0001 wandb.project='ATACOM_A1_Vel'
# python experiments/run_exp_a1.py atacom.enable=False control.type='Vel' control.action_scale=2.5 train.params.ent_coeff=0.0001 wandb.project='ATACOM_A1_Vel'

# python experiments/run_exp_a1.py constraints.foot_pos_max_z=-0.15 atacom.lambda_c_i=20. train.params.ent_coeff=1e-3
# python experiments/run_exp_a1.py constraints.foot_pos_max_z=-0.2
# python experiments/run_exp_a1.py constraints.foot_pos_max_z=-0.3
# python experiments/run_exp_a1.py constraints.foot_pos_min_z=-0.3 atacom.lambda_c_i=0.
# python experiments/run_exp_a1.py constraints.foot_pos_min_z=-0.35 atacom.lambda_c_i=0.
# python experiments/run_exp_a1.py constraints.foot_pos_min_z=-0.4 atacom.lambda_c_i=0.

# python experiments/run_exp_a1.py constraints.foot_pos_max_z=-0.1 train.params.ent_coeff=0.001 control.action_scale=2.5
# python experiments/run_exp_a1.py constraints.foot_pos_min_z=-0.15 train.params.ent_coeff=0.001 control.action_scale=2.5
# python experiments/run_exp_a1.py constraints.foot_pos_min_z=-0.2 train.params.ent_coeff=0.001 control.action_scale=2.5

# python experiments/run_exp_a1.py constraints.foot_pos_alpha=0.5 constraints.foot_pos_beta=0.5
# python experiments/run_exp_a1.py constraints.foot_pos_alpha=0.4 constraints.foot_pos_beta=0.4
# python experiments/run_exp_a1.py constraints.foot_pos_alpha=0.3 constraints.foot_pos_beta=0.3
# python experiments/run_exp_a1.py constraints.foot_pos_alpha=0.2 constraints.foot_pos_beta=0.2
# python experiments/run_exp_a1.py constraints.foot_pos_alpha=0.1 constraints.foot_pos_beta=0.1

#max_z = 0.4 missing

# python experiments/run_exp_a1.py experiment_subdir='Foot_rot_0.0' constraints.foot_rot_base='[[0., -0., 0.]]' constraints.foot_rot_min=0.8 #atacom.lambda_c_i=0
# python experiments/run_exp_a1.py experiment_subdir='Foot_rot_0.1' constraints.foot_rot_base='[[0., -0.1, 0.]]' constraints.foot_rot_min=0.8 #atacom.lambda_c_i=0

# python experiments/run_exp_a1.py experiment_subdir='Foot_rot_0.3_1.3_2_NI' seed=[1]  constraints.foot_rot_base='[[0., -0.3, 0.]]' constraints.foot_rot_min=1.3 constraints.foot_rot_max=2.  atacom.lambda_c_i=0
# python experiments/run_exp_a1.py experiment_subdir='Foot_rot_0.3_1.1_3.14_NI' seed=[1]  constraints.foot_rot_base='[[0., -0.3, 0.]]' constraints.foot_rot_min=1.1 constraints.foot_rot_max=3.14  atacom.lambda_c_i=0
# python experiments/run_exp_a1.py experiment_subdir='Foot_rot_0.3_1.1_2_NI' seed=[3]  constraints.foot_rot_base='[[0., -0.3, 0.]]' constraints.foot_rot_min=1.1 constraints.foot_rot_max=2  atacom.lambda_c_i=0
# python experiments/run_exp_a1.py experiment_subdir='Foot_rot_0.3_1.1_2_NI' seed=[4]  constraints.foot_rot_base='[[0., -0.3, 0.]]' constraints.foot_rot_min=1.1 constraints.foot_rot_max=2  atacom.lambda_c_i=0
# python experiments/run_exp_a1.py experiment_subdir='Foot_rot_0.3_1.1_2_NI' seed=[5]  constraints.foot_rot_base='[[0., -0.3, 0.]]' constraints.foot_rot_min=1.1 constraints.foot_rot_max=2  atacom.lambda_c_i=0

# python experiments/plot/plot_metric.py --cfg_paths /home/stud_magliano/projects/SafeLocomotion/logs/Foot_rot_0.3_1.1_2_NI
# python experiments/run_exp_a1.py experiment_subdir='Foot_rot_0.3_1.57_2.5_NI' seed=[1]  constraints.foot_rot_base='[[0., -0.3, 0.]]' constraints.foot_rot_min=1.57 constraints.foot_rot_max=2.5  atacom.lambda_c_i=0
# python experiments/plot/plot_metric.py --cfg_paths /home/stud_magliano/projects/SafeLocomotion/logs/Foot_rot_0.3_1.57_3.14_NI

# python experiments/run_exp_a1.py experiment_subdir='Joint_0.9_0.7_0.9' seed=[1] constraints.joint_percentage=true constraints.joint_limit='[0.9, 0.5, 0.9]'
# python experiments/run_exp_a1.py experiment_subdir='Joint_1.2_0.7_1.2_NI' seed=[4] constraints.joint_percentage=true constraints.joint_limit='[1.2, 0.7, 1.2]' atacom.lambda_c_i=0.
# python experiments/run_exp_a1.py experiment_subdir='Joint_1.2_0.7_1.2_NI' seed=[5] constraints.joint_percentage=true constraints.joint_limit='[1.2, 0.7, 1.2]' atacom.lambda_c_i=0.
# python experiments/run_exp_a1.py experiment_subdir='Joint_0.7_0.7_1_A_NI' seed=[1]  constraints.joint_limit='[0.7, 0.7, 1.]' atacom.lambda_c_i=0.
# python experiments/plot/plot_metric.py --cfg_paths /home/stud_magliano/projects/SafeLocomotion/trained_policy/atacom/Min_height_0.05_DC --compare_paths /home/stud_magliano/projects/SafeLocomotion/trained_policy/baseline/Velocity_baseline /home/stud_magliano/projects/SafeLocomotion/trained_policy/atacom/Min_height_0.05
# python experiments/plot/plot_metric.py --cfg_paths /home/stud_magliano/projects/SafeLocomotion/logs/Joint_1_0.7_1 --compare_paths /home/stud_magliano/projects/SafeLocomotion/trained_policy/baseline/Velocity_baseline

# python experiments/run_exp_a1.py experiment_subdir='Foot_min_height_0.15_NW_C20' seed=[5] constraints.foot_pos_max_z=-0.15 atacom.lambda_c_i=20. atacom.integral_window=0
# python experiments/plot/plot_metric.py --cfg_paths /home/stud_magliano/projects/SafeLocomotion/logs/Foot_min_height_0.15_NW_C20

# python experiments/run_exp_a1.py experiment_subdir='Max_height_0.1_DC' seed=[3]  constraints.foot_pos_min_z=-0.1 atacom.lambda_c_i=0.
# python experiments/run_exp_a1.py experiment_subdir='Foot_max_height_0.1_IN' seed=[5] constraints.foot_pos_min_z=-0.1 atacom.lambda_c_i=0.
# python experiments/plot/plot_metric.py --cfg_paths /home/stud_magliano/projects/SafeLocomotion/logs/Foot_max_h_0.15_NI_1 --compare_paths /home/stud_magliano/projects/SafeLocomotion/trained_policy/baseline/Velocity_baseline

# python experiments/run_exp_a1.py experiment_subdir='Foot_min_height_0.25_rot_0.3_1.57_3.14_NI' seed=[5]  constraints.foot_pos_max_z=-0.25 constraints.foot_rot_base='[[0., -0.3, 0]]' constraints.foot_rot_min=1.57 atacom.lambda_c_i=0.
# python experiments/run_exp_a1.py experiment_subdir='Foot_rot_0.3_1.57_2_DC_EC' seed=[1] constraints.foot_rot_base='[[0., -0.3, 0]]' constraints.foot_rot_min=1.57 constraints.foot_rot_max=2
# python experiments/plot/plot_metric.py --cfg_paths /home/stud_magliano/projects/SafeLocomotion/trained_policy/atacom/Foot_pos_0.2_DC --compare_paths /home/stud_magliano/projects/SafeLocomotion/trained_policy/baseline/Velocity_baseline
# python experiments/plot/plot_metric.py --cfg_paths /home/stud_magliano/projects/SafeLocomotion/trained_policy/atacom/Foot_rot_0.3_DC --compare_paths /home/stud_magliano/projects/SafeLocomotion/trained_policy/baseline/Velocity_baseline
# python experiments/plot/plot_metric.py --cfg_paths /home/stud_magliano/projects/SafeLocomotion/trained_policy/atacom/Joint_0.7_1_0.7_DC --compare_paths /home/stud_magliano/projects/SafeLocomotion/trained_policy/baseline/Velocity_baseline
# python experiments/plot/plot_metric.py --cfg_paths /home/stud_magliano/projects/SafeLocomotion/trained_policy/atacom/Max_height_0.15_DC --compare_paths /home/stud_magliano/projects/SafeLocomotion/trained_policy/baseline/Velocity_baseline
# python experiments/plot/plot_metric.py --cfg_paths /home/stud_magliano/projects/SafeLocomotion/trained_policy/atacom/Max_height_0.15_DC --compare_paths /home/stud_magliano/projects/SafeLocomotion/trained_policy/baseline/Velocity_baseline --epsilon 0.1
# python experiments/plot/plot_metric.py --cfg_paths /home/stud_magliano/projects/SafeLocomotion/trained_policy/atacom/Min_height_0.15_DC --compare_paths /home/stud_magliano/projects/SafeLocomotion/trained_policy/atacom/Min_height_0.15_DC_EC /home/stud_magliano/projects/SafeLocomotion/trained_policy/baseline/Velocity_baseline
# python experiments/plot/plot_metric.py --cfg_paths /home/stud_magliano/projects/SafeLocomotion/trained_policy/atacom/Min_height_0.05 --compare_paths /home/stud_magliano/projects/SafeLocomotion/trained_policy/atacom/Min_height_0.05_DC /home/stud_magliano/projects/SafeLocomotion/trained_policy/baseline/Velocity_baseline
python experiments/plot/plot_metric.py --cfg_paths /home/stud_magliano/projects/SafeLocomotion/trained_policy/atacom/Min_height_0.05  --compare_paths /home/stud_magliano/projects/SafeLocomotion/trained_policy/baseline/Velocity_baseline
# python experiments/plot/plot_metric.py --cfg_paths /home/stud_magliano/projects/SafeLocomotion/trained_policy/atacom/Min_height_0.05_DC --compare_paths /home/stud_magliano/projects/SafeLocomotion/trained_policy/baseline/Velocity_baseline
# python experiments/plot/plot_metric.py --cfg_paths /home/stud_magliano/projects/SafeLocomotion/trained_policy/atacom/Min_height_0.15_DC --compare_paths /home/stud_magliano/projects/SafeLocomotion/trained_policy/baseline/Velocity_baseline
# python experiments/plot/plot_metric.py --cfg_paths /home/stud_magliano/projects/SafeLocomotion/trained_policy/atacom/Min_height_0.15_DC_EC --compare_paths /home/stud_magliano/projects/SafeLocomotion/trained_policy/baseline/Velocity_baseline
# python experiments/plot/plot_metric.py --cfg_paths /home/stud_magliano/projects/SafeLocomotion/trained_policy/atacom/Min_height_0.25_DC
# python experiments/plot/plot_metric.py --cfg_paths /home/stud_magliano/projects/SafeLocomotion/trained_policy/atacom/Min_height_0.35_DC --compare_paths /home/stud_magliano/projects/SafeLocomotion/trained_policy/baseline/Velocity_baseline --epsilon 0.15
# python experiments/plot/plot_metric.py --cfg_paths /home/stud_magliano/projects/SafeLocomotion/trained_policy/baseline/Velocity_baseline  --compare_paths /home/stud_magliano/projects/SafeLocomotion/trained_policy/baseline/Position_baseline

# python /home/stud_magliano/projects/SafeLocomotion/experiments/lstsq/lstsq_comparison.py

# python experiments/run_exp_a1.py experiment_subdir='Min_height_0.05_DC_EC' seed=[4] constraints.foot_pos_max_z=-0.05
# python experiments/run_exp_a1.py experiment_subdir='Min_height_0.05_DC_EC' seed=[5] constraints.foot_pos_max_z=-0.05
# python experiments/run_exp_a1.py experiment_subdir='Min_height_0.35_DC' seed=[5] constraints.foot_pos_max_z=-0.35 atacom.lambda_c_i=0
# python experiments/run_exp_a1.py experiment_subdir='Min_height_0.05' seed=[4] constraints.foot_pos_max_z=-0.05 atacom.lambda_c_i=0
# python experiments/plot/plot_metric.py --cfg_paths /home/stud_magliano/projects/SafeLocomotion/logs/Min_height_0.05_DC_EC --compare_paths /home/stud_magliano/projects/SafeLocomotion/trained_policy/baseline/Velocity_baseline

# python experiments/run_exp_a1.py experiment_subdir='Foot_min_height_0.3_DC_EC' seed=[1] constraints.foot_pos_max_z=-0.3 atacom.lambda_c_i=0.1
# python experiments/run_exp_a1.py experiment_subdir='Foot_min_height_0.4_IN' seed=[5] constraints.foot_pos_max_z=-0.4 atacom.lambda_c_i=0
# python experiments/plot/plot_metric.py --cfg_paths /home/stud_magliano/projects/SafeLocomotion/logs/Foot_min_height_0.4_IN --compare_paths /home/stud_magliano/projects/SafeLocomotion/trained_policy/baseline/Velocity_baseline

# python experiments/run_exp_a1.py experiment_subdir='Foot_min_height_0.1_ND_NI' seed=[4] constraints.foot_pos_max_z=-0.1 atacom.lambda_c_i=0
# python experiments/run_exp_a1.py experiment_subdir='Foot_min_height_0.1_ND_NI' seed=[5] constraints.foot_pos_max_z=-0.1 atacom.lambda_c_i=0
# python experiments/plot/plot_metric.py --cfg_paths /home/stud_magliano/projects/SafeLocomotion/logs/Foot_min_height_0.1_ND_NI --compare_paths /home/stud_magliano/projects/SafeLocomotion/trained_policy/baseline/Velocity_baseline
# python experiments/run_exp_a1.py experiment_subdir='Foot_rot_0.0' seed=[3] constraints.foot_rot_base='[[0., 0., 0]]' constraints.foot_rot_min=0.8
# python experiments/run_exp_a1.py experiment_subdir='Foot_rot_0.0' seed=[4] constraints.foot_rot_base='[[0., 0., 0]]' constraints.foot_rot_min=0.8
# python experiments/run_exp_a1.py experiment_subdir='Foot_rot_0.0' seed=[5] constraints.foot_rot_base='[[0., 0., 0]]' constraints.foot_rot_min=0.8

# python experiments/run_exp_a1.py experiment_subdir='Foot_max_min_height_0.05_0.15_DC' seed=[4] constraints.foot_pos_max_z=-0.05 constraints.foot_pos_min_z=-0.15 atacom.lambda_c_i=0
# python experiments/run_exp_a1.py experiment_subdir='Foot_max_min_height_0.05_0.15_DC' seed=[5] constraints.foot_pos_max_z=-0.05 constraints.foot_pos_min_z=-0.15 atacom.lambda_c_i=0
# python experiments/run_exp_a1.py experiment_subdir='Foot_pos_0.2_DC' seed=[4] constraints.foot_pos_alpha=0.2 constraints.foot_pos_beta=0.2 atacom.lambda_c_i=0.
# python experiments/run_exp_a1.py experiment_subdir='Foot_pos_0.2_DC' seed=[5] constraints.foot_pos_alpha=0.2 constraints.foot_pos_beta=0.2 atacom.lambda_c_i=0.
# python experiments/plot/plot_metric.py --cfg_paths /home/stud_magliano/projects/SafeLocomotion/logs/Joint_1_0.7_1_NI --compare_paths /home/stud_magliano/projects/SafeLocomotion/trained_policy/baseline/Velocity_baseline

# python experiments/run_exp_a1.py test=True render=True record=True checkpoint=/home/stud_magliano/projects/SafeLocomotion/trained_policy/baseline/Position_baseline/A1_2025-05-16-15-46-57 constraints.joint_limit=[0.1] atacom.lambda_c_i=0.