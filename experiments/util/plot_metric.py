import matplotlib.pyplot as plt
import os
import torch

import sys
sys.path.append('/home/magliano/Project/SafeLocomotion')

from atacom.envs.costr_log_utils import get_constraint_info
from experiments.kinematics_a1 import LinkPos
from experiments.util.lie_group import SO3

def plot_metric(state, env_info, epoch, plot_path):
    constraint_epsilon = 0
    joint_pos = state[:, [ 6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]]

    joint_hist(joint_pos, env_info, epoch, plot_path)
    foot_pos_hist(joint_pos, env_info, epoch, plot_path)
    foot_rot_hist(joint_pos, env_info, epoch, plot_path)

def joint_hist(joint_pos, env_info, epoch, plot_path):
    feet_names = ['FL', 'FR', 'RL', 'RR']
    joint_names = ['Hip', 'Thigh', 'Calf']

    # Plot histogram of joint angles positions
    fig, axs = plt.subplots(4, 3, figsize=(15, 20))
    for i in range(4):
        for j in range(3):
            axs[i, j].hist(joint_pos[:, i * 3 + j], bins=100)
            axs[i, j].set_title(f"{feet_names[i]} {joint_names[j]}")

    os.makedirs(f"{plot_path}/distribution/epoch_{epoch}", exist_ok=True)
    plt.savefig(f"{plot_path}/distribution/epoch_{epoch}/joint_pos_distribution_{epoch}.png")


def foot_pos_hist(joint_pos, env_info, epoch, plot_path):
    feet_names = ['FL', 'FR', 'RL', 'RR']
    axes = ['x', 'y', 'z']

    # Plot histogram of foot positions
    feet = [LinkPos(env_info['urdf_path'], side + '_foot', side + '_thigh', env_info['default_joint_pos'],  env_info['action']['idx'][side]) for side in feet_names]

    fig, axs = plt.subplots(4, 3, figsize=(15, 20))
    for i in range(4):
        for j in range(3):
            foot = feet[i]
            foot_pos = foot.get_pos(joint_pos.cuda() + env_info['default_joint_pos']).cpu()
            axs[i, j].hist(foot_pos[:, j], bins=100)
            axs[i, j].set_title(f"{feet_names[i]} {axes[j]}")

    os.makedirs(f"{plot_path}/distribution/epoch_{epoch}", exist_ok=True)
    plt.savefig(f"{plot_path}/distribution/epoch_{epoch}/foot_pos_distribution_{epoch}.png")

def foot_rot_hist(joint_pos, env_info, epoch, plot_path):
    feet_names = ['FL', 'FR', 'RL', 'RR']

    feet = [LinkPos(env_info['urdf_path'], side + '_foot', side + '_thigh', env_info['default_joint_pos'],  env_info['action']['idx'][side]) for side in feet_names]

    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    for i in range(2):
        for j in range(2):
            foot = feet[i * 2 + j]
            foot_rot = foot.get_rot(joint_pos.cuda() + env_info['default_joint_pos']).cpu()
            foot_angle = torch.norm(SO3.Log(foot_rot), dim=1)
            axs[i, j].hist(foot_angle, bins=100)
            axs[i, j].set_title(f"{feet_names[i * 2 + j]}")

    os.makedirs(f"{plot_path}/distribution/epoch_{epoch}", exist_ok=True)
    plt.savefig(f"{plot_path}/distribution/epoch_{epoch}/foot_rot_distribution_{epoch}.png")

def save_dataset(dataset, dataset_path, epoch):
    os.makedirs(f"{dataset_path}/epoch_{epoch}", exist_ok=True)
    torch.save(dataset.state.cpu(), f"{dataset_path}/epoch_{epoch}/state_{epoch}.pt")
    for key in dataset.info.keys():
        if 'constraint' in key:
            torch.save(dataset.info[key], f"{dataset_path}/epoch_{epoch}/{key}_{epoch}.pt")

def load_dataset(dataset_path, num_epoch):
    dataset = []
    for epoch in range(num_epoch):
        dataset_dict = {}
        for file_name in os.listdir(f"{dataset_path}/epoch_{epoch}"):
            key = '_'.join(file_name.split('_')[:-1])
            data = torch.load(f"{dataset_path}/epoch_{epoch}/{file_name}")
            
            dataset_dict[key] = data
        dataset.append(dataset_dict)
        
    return dataset

def plot_constraint(dataset, num_epoch, plot_path, deep, epsilon=0):
    os.makedirs(f'{plot_path}/constraint', exist_ok=True)
    constraints = {}
    for epoch in range(num_epoch):
        for key in dataset[epoch].keys():
            info = get_constraint_info({key: dataset[epoch][key]}, deep=deep, full=True, epsilon=epsilon)
            if info != {}:
                constr_key = '_'.join(key.split('_')[:-1])
                if constr_key not in constraints:
                    constraints[constr_key] = {}
                info_dict = {}
                for k1 in info.keys():
                    for k2 in info[k1].keys():
                        if k2 not in info_dict:
                            info_dict[k2] = []
                        info_dict[k2].append(info[k1][k2])
                for k in info_dict.keys():
                    if k not in constraints[constr_key]:
                        constraints[constr_key][k] = torch.tensor(info_dict[k]).unsqueeze(0)
                    else:
                        constraints[constr_key][k] = torch.cat([constraints[constr_key][k], torch.tensor(info_dict[k]).unsqueeze(0)], dim=0)

    for key in constraints.keys():  
        constraint = constraints[key]
        for k in constraint.keys():
            constraint_plot(constraint[k], constraint[k].shape[1], f'{plot_path}/constraint', key, f'{k}{"_deep" if deep else ""}{"_e" + str(epsilon) if epsilon > 0 else ""}')             
            
def constraint_plot(constraint, num_constraint, plot_path, dir_path, plot_name, fig=None, axs=None):
    if fig is None or axs is None:
        fig, axs = subplot(num_constraint)
     
    for i in range(num_constraint):
        axs[i].plot(constraint[:, i])
        axs[i].set_title(f"Constraint {i}")
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel('Violation')
        axs[i].grid()

    fig.suptitle(plot_name.capitalize())
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    os.makedirs(f"{plot_path}/{dir_path}", exist_ok=True)
    plt.savefig(f"{plot_path}/{dir_path}/{plot_name}.png")
    plt.close(fig)

def plot_experiment_metric(dataset_path, plot_path, num_epoch):
    epsilon = 0
    dataset = load_dataset(dataset_path, num_epoch)
    plot_constraint(dataset, num_epoch, plot_path, deep=True, epsilon=epsilon)
    plot_constraint(dataset, num_epoch, plot_path, deep=False, epsilon=epsilon)

def subplot(num):
    n_col = 3 if num > 3 else num
    n_row = num // n_col + (num % n_col > 0)
    fig, axs = plt.subplots(n_row, n_col, figsize=(3*(n_col + 1), 3*(n_row + 1)))
    axs = axs.flatten() if num > 1 else [axs]
    return fig, axs
    


    