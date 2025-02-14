import torch
import numpy as np
import scipy
import matplotlib.pyplot as plt
import tqdm
import itertools
from math import ceil
import os

import hydra
from omegaconf import DictConfig
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict
from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *

from atacom import VelocityControlSystem
from atacom import ConstraintList
from atacom import ATACOMController

from atacom.envs.a1 import A1Env
from experiments.atacom_a1 import FootPosConstraint, JointPosConstraint

@hydra.main(config_name="config", config_path="../cfg", version_base="1.1")
def main(cfg: DictConfig):

    cfg_dict = omegaconf_to_dict(cfg)
    cfg_dict['num_envs'] = 1

    env, env_info, atacom_controller = get_atacom(cfg_dict)

    n_points = 20
    joint_idx = [0, 1, 2]
    constr_idx = [0, 1, 2]

    limit_linspace = [torch.linspace(env_info['joint_pos_limit'][0][i], env_info['joint_pos_limit'][1][i], n_points) for i in joint_idx]
    # Joint pos   
    q_val = torch.tensor(list(itertools.product(*limit_linspace))).to('cuda')
    q = torch.zeros((q_val.shape[0], env_info['n_joints'])).to('cuda')
    q[:, joint_idx] = q_val

    q = q.unsqueeze(1).repeat(1, cfg_dict['num_envs'], 1)

    xyz, k_val, J_inv, slack = [], [], [], []

    for i in tqdm.tqdm(range(q.shape[0])):
        pos = env_info['fun']['get_relative_link'](q[i], 'FL_foot')[:, :3, 3]
        k = atacom_controller.constraints.k(q[i])
        mu = atacom_controller.get_mu(k)
        Ju = atacom_controller.J_u(atacom_controller.J_G(q[i], None), mu)

        xyz.append(pos)
        k_val.append(k)
        J_inv.append(-torch.pinverse(Ju) * (k + mu))
        slack.append(mu)

    q = q.cpu().numpy()
    xyz = torch.stack(xyz, dim=0).cpu().numpy()
    k_val = torch.stack(k_val, dim=0).cpu().numpy()
    J_inv = torch.stack(J_inv, dim=0).cpu().numpy()
    slack = torch.stack(slack, dim=0).cpu().numpy()

    # constraint_plot(q[:, 0, :3].squeeze(), k_val[:, 0].squeeze(), J_inv[:, 0, :3].squeeze(), slack[:, 0].squeeze())
    contour_plot(q[:, 0, joint_idx], k_val[:, 0, constr_idx], J_inv[:, 0, joint_idx][:, :, constr_idx], name='FL_foot')
    # contour_plot(q[:, 0, joint_idx], xyz[:, 0, constr_idx], name='FL_foot_pos')

def get_atacom(cfg_dict):

    env, env_info = A1Env.build_env(cfg_dict)

    dyn = VelocityControlSystem(dim_q=env_info['n_joints'], vel_limit=env_info['joint_vel_limit'][1])
    constr_list = ConstraintList(dim_q=env_info['n_joints'])

    constr_pos = FootPosConstraint(env_info['n_joints'], 'FL', env_info['fun']['get_relative_link'], env_info['fun']['get_J_relative_link'], alpha=0.2, beta=0.4, min_z=-0.1, max_z=0.1)
    constr_list.add_constraint(constr_pos)

    perc = 0.2
    limit = torch.vstack([torch.tensor([(high + low) / 2 - perc * (high - low) / 2 for low, high in zip(env_info['joint_pos_limit'][0], env_info['joint_pos_limit'][1])]).to('cuda'), torch.tensor([(high + low) / 2 + perc * (high - low) / 2 for low, high in zip(env_info['joint_pos_limit'][0], env_info['joint_pos_limit'][1])]).to('cuda')])
    # constr_joint = JointPosConstraint(env_info['n_joints'], limit)
    # constr_list.add_constraint(constr_joint)


    cfg_dict['atacom']['slack_beta'] = torch.tensor(cfg_dict['atacom']['slack_beta'])
    cfg_dict['atacom']['lambda_c'] = 0.5 / env.dt

    atacom_controller = ATACOMController(constr_list, dyn,
                                        slack_beta=cfg_dict['atacom']['slack_beta'],
                                        slack_tol=cfg_dict['atacom']['slack_tol'],
                                        slack_dynamics_type=cfg_dict['atacom']['slack_dynamics_type'],
                                        drift_compensation_type=cfg_dict['atacom']['drift_compensation_type'],
                                        drift_clipping=cfg_dict['atacom']['drift_clipping'],
                                        lambda_c=cfg_dict['atacom']['lambda_c'])
    
    return env, env_info, atacom_controller

def constraint_plot(q, k_val, J_inv, slack, num_col=3):
    row = ceil(q.shape[-1] / num_col)
    col = min(q.shape[-1], num_col)

    fig, axs = plt.subplots(row, col, figsize=(5*col, 5*row))

    for i in range(row):
        for j in range(col):
            if i * col + j < q.shape[-1]:
                ax = axs[i, j] if row > 1 else axs[j]
                ax.plot(q[:, i * col + j], k_val, label='k')
                ax.plot(q[:, i * col + j], slack, label='slack')
                ax.legend()
                ax.set_title(f'Joint {i * col + j}')
                ax.set_xlabel('Joint pos')
                ax.set_ylabel('Constraint value')

    fig.savefig('plot/constraint/constraint_plot.png')

def contour_plot(q, k_val, J=None, name=None):
    comb_idx = list(itertools.combinations(range(q.shape[-1]), 2))  
    row = k_val.shape[-1] 
    col = len(comb_idx)
    fig, axs = plt.subplots(row, col, figsize=(5*col, 5*row))
    
    for i in range(row):
        if J is not None:
            q_pair, k_pair, J_pair = get_pair(q, k_val[:, i], J[:, :, i]) 
        else:
            q_pair, k_pair = get_pair(q, k_val[:, i])
        for j in range(col):
            ax = axs.flatten()[i * col + j] if row > 1 or col > 1 else axs
            contour(ax, q_pair[j][:, 0], q_pair[j][:, 1], k_pair[j])
            if J is not None:
                ax.quiver(q_pair[j][:, 0], q_pair[j][:, 1], J_pair[j][:, 0], J_pair[j][:, 1])
            ax.set_xlabel(f'Joint {comb_idx[j][0]}')
            ax.set_ylabel(f'Joint {comb_idx[j][1]}')

    fig.suptitle('Constraint value')
    fig.savefig(f'plot/constraint/{name + "_" if name else ""}contour_plot.png')

def contour(ax, x, y, val):
    n_x = np.unique(x).shape[0]
    n_y = np.unique(y).shape[0]
    X = x.reshape(n_x, n_y)
    Y = y.reshape(n_x, n_y)
    Z = val.reshape(n_x, n_y)

    contour = ax.contour(X, Y, Z)
    ax.clabel(contour, inline=True, fontsize=8)

def get_pair(*arrays):
    x = arrays[0]
    assert all([x.shape[0] == arr.shape[0] for arr in arrays])
    x_unique = [np.unique(x[:, i]).shape[0] for i in range(x.shape[-1])]
    assert all([x_unique[0] == x_unique[i] for i in range(1, x.shape[-1])])
    x_unique = x_unique[0]

    comb_idx = list(itertools.combinations(range(x.shape[-1]), 2))

    arrays_pair = []
    for arr in arrays:
        pairs = []
        for i, j in comb_idx:
            if arr.ndim == 1:
                pairs.append(arr[get_indexs(i, j, x.shape[-1], x_unique)])
            else:
                assert arr.shape == x.shape
                pairs.append(arr[get_indexs(i, j, x.shape[-1], x_unique)][:, (i, j)])
        arrays_pair.append(pairs)
    
    return arrays_pair if len(arrays_pair) > 1 else arrays_pair[0]

def get_indexs(i, j, n, v):
    mask = np.ones(v ** n, dtype=bool)
    index = np.array(list(itertools.product(range(v), repeat=n)))
    for d in range (n):
        if d != i and d != j:
            mask &= (index[:, d] == ceil(v / 2))

    return mask

if __name__ == '__main__':
    main()

