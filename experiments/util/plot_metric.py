import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
import os
import torch
import numpy as np

import sys
sys.path.append('/home/magliano/Project/SafeLocomotion')

from mushroom_rl.utils.plot import get_mean_and_confidence

from atacom.envs.costr_log_utils import get_constraint_info, epsilon_distribution
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
    feet_base_rot = [[0., -0.3, 0.], [0., -0.3, 0.], [0., -0.3, 0.], [0., -0.3, 0.]]

    feet = [LinkPos(env_info['urdf_path'], side + '_foot', side + '_thigh', env_info['default_joint_pos'],  env_info['action']['idx'][side]) for side in feet_names]
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    rots = []
    for i in range(2):
        for j in range(2):
            foot = feet[i * 2 + j]
            foot.set_base_rot(SO3.Exp(torch.tensor(feet_base_rot[i * 2 + j]).cuda()))
            foot_rot = foot.get_rot(joint_pos.cuda() + env_info['default_joint_pos']).cpu()
            rots.append(foot_rot)
            foot_angle = torch.norm(SO3.Log(foot_rot), dim=1)
            axs[i, j].hist(foot_angle, bins=100)
            axs[i, j].set_title(f"{feet_names[i * 2 + j]}")

    os.makedirs(f"{plot_path}/distribution/epoch_{epoch}", exist_ok=True)
    plt.savefig(f"{plot_path}/distribution/epoch_{epoch}/foot_rot_distribution_{epoch}.png")

def save_dataset(dataset, dataset_path, epoch):
    os.makedirs(f"{dataset_path}/epoch_{epoch}", exist_ok=True)
    torch.save(dataset.state.cpu(), f"{dataset_path}/epoch_{epoch}/state_{epoch}.pt")
    constr = None
    for key in dataset.info.keys():
        if 'constraint' in key:
            torch.save(dataset.info[key], f"{dataset_path}/epoch_{epoch}/{key}_{epoch}.pt")
            constr = torch.cat([constr, dataset.info[key]], dim=-1) if constr is not None else dataset.info[key]
    if constr is not None:
        torch.save(constr.cpu(), f"{dataset_path}/epoch_{epoch}/constraint_{epoch}.pt")

def save_metric(J, R, E, V, task_info, metric_path, epoch):
    os.makedirs(f"{metric_path}/epoch_{epoch}", exist_ok=True)
    torch.save(torch.tensor(J).cpu(), f"{metric_path}/epoch_{epoch}/J_{epoch}.pt")
    torch.save(torch.tensor(R).cpu(), f"{metric_path}/epoch_{epoch}/R_{epoch}.pt")
    torch.save(torch.tensor(E).cpu(), f"{metric_path}/epoch_{epoch}/E_{epoch}.pt")
    torch.save(torch.tensor(V).cpu(), f"{metric_path}/epoch_{epoch}/V_{epoch}.pt")
    for k_type, v_type in task_info.items():
        if k_type != 'constraint':
            if isinstance(v_type, dict):
                for k_name, v_name in v_type.items():
                    if isinstance(v_name, dict):
                        for k_metric, v_metric in v_name.items():
                            torch.save(torch.tensor(v_metric).cpu(), f"{metric_path}/epoch_{epoch}/{k_type}_{k_name}_{k_metric}_{epoch}.pt")
                    else:
                        torch.save(torch.tensor(v_name).cpu(), f"{metric_path}/epoch_{epoch}/{k_type}_{k_name}_{epoch}.pt")
            else:
                torch.save(torch.tensor(v_type).cpu(), f"{metric_path}/epoch_{epoch}/{k_type}_{epoch}.pt")

def load_dataset(dataset_path, num_epoch):
    dataset = []
    for epoch in range(num_epoch):
        dataset_dict = {}
        for file_name in os.listdir(f"{dataset_path}/epoch_{epoch}"):
            key = '_'.join(file_name.split('_')[:-1]) if '_'.join(file_name.split('_')[:-1]) else file_name
            data = torch.load(f"{dataset_path}/epoch_{epoch}/{file_name}", weights_only=True)

            dataset_dict[key] = data
        dataset.append(dataset_dict)

    return dataset

def load_metric(metric_path, num_epoch):
    metric_dict = {}
    for epoch in range(num_epoch):
        for file_name in os.listdir(f"{metric_path}/epoch_{epoch}"):
            key = '_'.join(file_name.split('_')[:-1])
            data = torch.load(f"{metric_path}/epoch_{epoch}/{file_name}", weights_only=True)

            if key not in metric_dict:
                metric_dict[key] = data.unsqueeze(0)
            else:
                metric_dict[key] = torch.cat([metric_dict[key], data.unsqueeze(0)], dim=0)

    return metric_dict

def constraints_from_dataset(dataset, num_epoch, deep, epsilon_values, epsilon=0):
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
                if epoch == num_epoch - 1:
                    epsilon_info = epsilon_distribution({key: dataset[epoch][key]}, eps_points=epsilon_values)[key]
                    constraints[constr_key]['epsilon'] = epsilon_info
    return constraints

def plot_constraint(constraint_dicts, plot_path, labels, epsilon_values, deep=True, epsilon=0):
    for key in common_keys(constraint_dicts):

        for k in constraint_dicts[0][0][key].keys():
            fig, axs = subplot(constraint_dicts[0][0][key][k].shape[-1])
            for i, constraint_dicts_exp in enumerate(constraint_dicts):
                values = torch.stack([constraint_dict[key][k] for constraint_dict in constraint_dicts_exp], dim=0)
                if 'epsilon' not in k:
                    data_plot(values, values.shape[-1], f'{plot_path}/constraint', key, f'{k}{"_deep" if deep else ""}{"_e" + str(epsilon) if epsilon > 0 else ""}', axs=axs, enable_conf=False, ylabel=k.capitalize().replace("_", " "))
                else:
                    data_plot(values, values.shape[-1], f'{plot_path}/constraint', key, f'{k}{"_deep" if deep else ""}{"_e" + str(epsilon) if epsilon > 0 else ""}', axs=axs, enable_conf=False, xlabel='Violation value', ylabel='Percentage', x_values=epsilon_values[:-1])
            save_fig(fig, f'{plot_path}/constraint', key, f'{k}{"_deep" if deep else ""}{"_e" + str(epsilon) if epsilon > 0 else ""}', labels)

def common_keys(constraint_dicts):
    common_keys = set(constraint_dicts[0][0].keys())
    for constraint_dicts_exp in constraint_dicts:
        for constraint_dict in constraint_dicts_exp:
            common_keys.intersection_update(set(constraint_dict.keys()))
    return list(common_keys)

def data_plot(data, num_data, plot_path, dir_path, plot_name, x_values=None, fig=None, axs=None, ratio=1., enable_conf=True, xlabel='Epoch', ylabel='Value'):
    if fig is None and axs is None:
        fig, axs = subplot(num_data, ratio=ratio)

    for i in range(num_data):
        axs[i].minorticks_on()
        axs[i].set_axisbelow(True)
        axs[i].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        axs[i].grid(which='minor', linestyle=':', linewidth=0.4, alpha=0.5)

        if data.shape[0] > 1 and enable_conf:
            mean, conf = get_mean_and_confidence(np.array(data[:, :, i])) if num_data > 1 else get_mean_and_confidence(np.array(data))
            if np.isnan(conf).any():
                conf = np.zeros_like(mean)
        else:
            mean = data[:, :, i].mean(dim=0) if num_data > 1 else data.mean(dim=0)
            conf = torch.zeros_like(mean)

        if x_values is not None:
            axs[i].plot(x_values, mean, alpha=0.8)
            axs[i].fill_between(x_values.squeeze(), (mean - conf).squeeze(), (mean + conf).squeeze(), alpha=0.2)
        else:
            axs[i].plot(mean, alpha=0.8)
            axs[i].fill_between(range(mean.shape[0]), (mean - conf).squeeze(), (mean + conf).squeeze(), alpha=0.2)

        axs[i].set_xlabel(xlabel)
        if 'average' in plot_name or 'percentage' in plot_name:
            axs[i].set_ylabel(f'{ylabel} [%]')
        else:
            axs[i].set_ylabel(ylabel)

        # axs[i].lines[0].set_color('purple')

    if fig:
        save_fig(fig, plot_path, dir_path, plot_name)

def save_fig(fig, plot_path, dir_path, plot_name, labels):
    # fig.suptitle(plot_name.capitalize())
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    if 'epsilon' in plot_name:
        fig.legend(fig.axes[0].get_lines(), labels, loc='upper right', bbox_to_anchor=(0.95, 0.89))
    elif 'constraint' in dir_path or 'constraint' in plot_path:
        fig.legend(fig.axes[0].get_lines(), labels, loc='lower left', bbox_to_anchor=(0.15, 0.11))
    else:
        fig.legend(fig.axes[0].get_lines(), labels, loc='lower right', bbox_to_anchor=(0.95, 0.11))
    os.makedirs(f"{plot_path}/{dir_path}", exist_ok=True)
    fig.savefig(f"{plot_path}/{dir_path}/{plot_name}.png")
    plt.close(fig)

def plot_experiment_metric(base_paths, num_epoch, joint_pos_idx, default_joint_pos, constr_list=None, plot_path='plot', dataset_path='dataset', metric_path='metric', epsilon=0):
    exp_path = "/".join(str(base_paths[0][0]).split('/')[:-1])
    create_constraint_data(constr_list, base_paths, num_epoch, dataset_path, joint_pos_idx, default_joint_pos)
    epsilon_values = torch.linspace(0, 0.3, 30).unsqueeze(-1)

    # constraint_dicts_deep = [[constraints_from_dataset(load_dataset(f'{base}/{dataset_path}', num_epoch), num_epoch, True, epsilon) for base in base_exp] for base_exp in base_paths]
    constraint_dicts = [[constraints_from_dataset(load_dataset(f'{base}/{dataset_path}', num_epoch), num_epoch, False, epsilon_values, epsilon) for base in base_exp] for base_exp in base_paths]
    metric_dicts = [[load_metric(f'{base}/{metric_path}', num_epoch) for base in base_exp] for base_exp in base_paths]

    # plot_constraint(constraint_dicts_deep, f'{exp_path}/{plot_path}', labels=[str(base_paths[i][0]).split('/')[-2] for i in range(len(base_paths))], deep=True, epsilon=epsilon)
    plot_constraint(constraint_dicts, f'{exp_path}/{plot_path}', labels=[str(base_paths[i][0]).split('/')[-2] for i in range(len(base_paths))], epsilon_values=epsilon_values, deep=False, epsilon=epsilon)
    for k in metric_dicts[0][0].keys():
        if 'constraint' not in k:
            fig, axs = subplot(1, ratio=1.5)
            for i, metric_dicts_exp in enumerate(metric_dicts):
                values = torch.stack([metric_dict[k] for metric_dict in metric_dicts_exp], dim=0)
                data_plot(values, 1, f'{exp_path}/{plot_path}', 'metric', k, axs=axs, ylabel=k if k != 'R' else 'Reward')
            save_fig(fig, f'{exp_path}/{plot_path}', 'metric', k, [str(base_paths[i][0]).split('/')[-2] for i in range(len(base_paths))])

def create_constraint_data(constr_list, base_paths, epochs, dataset_path, joint_pos_idx, default_joint_pos):
    def _unwrap_state(obs):
        return obs[:, joint_pos_idx.to(obs.device)] + default_joint_pos.to(obs.device)
    if constr_list.constraints:
        for exp_paths in base_paths:
            for path in exp_paths:
                dataset = load_dataset(f'{path}/{dataset_path}', epochs)
                for i in range(epochs):
                    os.makedirs(f"{path}/{dataset_path}/epoch_{i}", exist_ok=True)
                    k = constr_list.k(_unwrap_state(dataset[i]['state'].to('cuda')))
                    torch.save(k.to('cpu'), f"{path}/{dataset_path}/epoch_{i}/constraint_{i}.pt")
                    for constr in constr_list.constraints:
                        k = constr.k_(_unwrap_state(dataset[i]['state'].to('cuda')))
                        torch.save(k.to('cpu'), f"{path}/{dataset_path}/epoch_{i}/{constr.name}_constraint_{i}.pt")

def subplot(num, ratio=1.):
    n_col = 3 if num > 3 else num
    n_row = num // n_col + (num % n_col > 0)
    fig, axs = plt.subplots(n_row, n_col, figsize=(3*(n_col + 1)*ratio, 3*(n_row + 1)))
    axs = axs.flatten() if num > 1 else [axs]
    return fig, axs
