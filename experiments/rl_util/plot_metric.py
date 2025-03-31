import matplotlib.pyplot as plt
import os
import torch

from atacom.envs.costr_log_utils import get_constraint_info
from experiments.kinematics_a1 import LinkPos
from experiments.util.plotter import Plotter

def plot_metric(state, env_info, epoch, plot_path):
    constraint_epsilon = 0
    joint_pos = state[:, [ 6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]]

    joint_hist(joint_pos, env_info, epoch, plot_path)
    foot_hist(joint_pos, env_info, epoch, plot_path)

def joint_hist(joint_pos, env_info, epoch, plot_path):
    feet_names = ['FL', 'FR', 'RL', 'RR']
    joint_names = ['Hip', 'Thigh', 'Calf']

    # Plot histogram of joint angles positions
    fig, axs = plt.subplots(4, 3, figsize=(15, 20))
    for i in range(4):
        for j in range(3):
            axs[i, j].hist(joint_pos[:, i * 3 + j], bins=100)
            axs[i, j].set_title(f"{feet_names[i]} {joint_names[j]}")

    os.makedirs(f"{plot_path}/distribution", exist_ok=True)
    plt.savefig(f"{plot_path}/distribution/joint_pos_distribution_{epoch}.png")


def foot_hist(joint_pos, env_info, epoch, plot_path):
    feet_names = ['FL', 'FR', 'RL', 'RR']
    axes = ['x', 'y', 'z']

    # Plot histogram of foot positions
    if env_info is not None:
        feet = [LinkPos(env_info['urdf_path'], side + '_foot', side + '_thigh', env_info['default_joint_pos'],  env_info['action']['idx'][side]) for side in feet_names]

        fig, axs = plt.subplots(4, 3, figsize=(15, 20))
        for i in range(4):
            for j in range(3):
                foot = feet[i]
                foot_pos = foot.get_pos(joint_pos.cuda() + env_info['default_joint_pos']).cpu()
                axs[i, j].hist(foot_pos[:, j], bins=100)
                axs[i, j].set_title(f"{feet_names[i]} {axes[j]}")

        os.makedirs(f"{plot_path}/distribution", exist_ok=True)
        plt.savefig(f"{plot_path}/distribution/feet_pos_distribution_{epoch if epoch is not None else ''}.png")

def save_dataset(dataset, dataset_path, epoch):
    os.makedirs(f"{dataset_path}/epoch_{epoch}", exist_ok=True)
    torch.save(dataset.state.cpu(), f"{dataset_path}/epoch_{epoch}/state_{epoch}.pt")
    for key in dataset.info.keys():
        if 'constraint' in key:
            torch.save(dataset.info[key], f"{dataset_path}/epoch_{epoch}/{key}_{epoch}.pt")

def load_dataset(dataset_path, num_epoch):
    pass

def plot_experiment_metric(dataset_path, plot_path, num_epoch):
    pass