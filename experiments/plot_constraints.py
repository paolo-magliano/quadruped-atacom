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
from atacom_a1 import FootPosConstraint, JointPosConstraint

@hydra.main(config_name="config", config_path="../cfg", version_base="1.1")
def main(cfg: DictConfig):

    cfg_dict = omegaconf_to_dict(cfg)
    cfg_dict['num_envs'] = 1
    cfg_dict['headless'] = True

    env, env_info, atacom_controller = get_atacom(cfg_dict)

    j_inv = True

    n_points = 20
    joint_idx = [0, 1, 2]
    constr_idx = [0, 1, 2]
    slack_linspace = [torch.linspace(0.1, 1., n_points)]

    if len(slack_linspace) > 0:
        joint_comb = [(j, None) for j in joint_idx]
    else:
        joint_comb = list(itertools.combinations(joint_idx, 2)) 

    q_pairs, xyz_pairs, k_val_pairs, J_inv_pairs, slack_pairs, m_val_pairs, slack_axis_pairs = [], [], [], [], [], [], []

    for i, j in joint_comb:
        idx = [x for x in [i, j] if x is not None]

        limit_linspace = [torch.linspace(env_info['joint_pos_limit'][0][x], env_info['joint_pos_limit'][1][x], n_points) for x in idx]
        # Joint pos   
        q_slack = torch.tensor(list(itertools.product(*[*slack_linspace, *limit_linspace]))).to('cuda')
        q_val = q_slack[:, len(slack_linspace):]
        slack_val = q_slack[:, :len(slack_linspace)]  
        q = ((env_info['joint_pos_limit'][0] + env_info['joint_pos_limit'][1]) / 2).clone().to('cuda').unsqueeze(0).repeat(q_val.shape[0], 1)
        q[:, idx] = q_val

        q = q.unsqueeze(1).repeat(1, cfg_dict['num_envs'], 1)
        slack_val = slack_val.unsqueeze(1).repeat(1, cfg_dict['num_envs'], len(constr_idx))

        xyz, k_val, J_inv, slack, m_val = [], [], [], [], []

        for i in tqdm.tqdm(range(q.shape[0])):
            pos = env_info['fun']['get_relative_link'](q[i], 'FL_foot')[:, :3, 3]
            k = atacom_controller.constraints.k(q[i])
            if len(slack_linspace) > 0:
                mu = slack_val[i]
            else:
                mu = atacom_controller.get_mu(k)
            A_mu = atacom_controller.slack.alpha(mu)
            Ju = atacom_controller.constraints.J_q(q[i], None)
            # Ju = atacom_controller.J_u(atacom_controller.J_G(q[i], None), mu)

            xyz.append(pos)
            k_val.append(k)
            if j_inv:
                J_inv.append(-torch.linalg.pinv(Ju) @ (mu + k)[:, :, None])
            else:
                J_inv.append(Ju)

            slack.append(mu)
            m_val.append(k + torch.diagonal(A_mu, dim1=1, dim2=2))

        q = q.cpu().numpy()[:, 0, idx]

        xyz = torch.stack(xyz, dim=0).cpu().numpy()[:, 0, constr_idx]
        k_val = torch.stack(k_val, dim=0).cpu().numpy()[:, 0, constr_idx]
        if len(slack_linspace) > 0:
            if j_inv:
                J_inv_s = [torch.stack(J_inv, dim=0)[:, 0, idx + [env_info['n_joints'] + c]][:, :, [c]] for c in constr_idx]
                J_inv = torch.cat(J_inv_s, dim=-1).cpu().numpy()
            else:
                J_inv_s = [torch.stack(J_inv, dim=0).cpu()[:, 0, [c]][:, :, idx + [env_info['n_joints'] + c]] for c in constr_idx]
                J_inv = torch.cat(J_inv_s, dim=-2).numpy()
        else:
            if j_inv:
                J_inv = torch.stack(J_inv, dim=0).cpu().numpy()[:, 0, idx][:, :, constr_idx]
            else:
                J_inv = torch.stack(J_inv, dim=0).cpu().numpy()[:, 0, constr_idx][:, :, idx]
        slack = torch.stack(slack, dim=0).cpu().numpy()[:, 0, constr_idx]
        m_val = torch.stack(m_val, dim=0).cpu().numpy()[:, 0, constr_idx]
        slack_axis = np.hstack([q, slack_val.cpu().numpy()[:, 0, :1]])

        q_pairs.append(q.copy())
        xyz_pairs.append(xyz.copy())
        k_val_pairs.append(k_val.copy())
        J_inv_pairs.append(J_inv.copy())
        slack_pairs.append(slack.copy())
        m_val_pairs.append(m_val.copy())
        slack_axis_pairs.append(slack_axis.copy())

    # constraint_plot(q_pairs, k_val_pairs, slack_pairs)
    if len(slack_linspace) > 0:
        contour_plot(joint_comb, slack_axis_pairs, m_val_pairs, J_inv_pairs, name='FL_foot_slack', j_inv=j_inv)
    else:
        contour_plot(joint_comb, q_pairs, k_val_pairs, J_inv_pairs, name='FL_foot', j_inv=j_inv)
    # contour_plot(q[:, 0, joint_idx], xyz[:, 0, constr_idx], name='FL_foot_pos')

def get_atacom(cfg_dict):

    env, env_info = A1Env.build_env(cfg_dict)

    dyn = VelocityControlSystem(dim_q=env_info['n_joints'], vel_limit=env_info['joint_vel_limit'][1])
    constr_list = ConstraintList(dim_q=env_info['n_joints'])

    constr_pos = FootPosConstraint(env_info['n_joints'], 'FL', env_info['fun']['get_relative_link'], env_info['fun']['get_J_relative_link'], env_info['fun']['get_commands'], alpha=0.2, beta=0.4, min_z=-0.1, max_z=0.1)
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

def constraint_plot(q, k_val=None, slack=None):
    row = k_val[0].shape[-1] if k_val is not None else slack[0].shape[-1]
    col = len(q)

    fig, axs = plt.subplots(row, col, figsize=(5*col, 5*row))

    for i in range(row):
        for j in range(col):
            ax = axs.flatten()[i * col + j] if row > 1 or col > 1 else axs
            if k_val is not None:
                ax.plot(q[j], k_val[j][:, i], label='k')
            if slack is not None:
                ax.plot(q[j], slack[j][:, i], label='slack')
            ax.legend()
            if i == 0:
                ax.set_title(f'Joint {j}')
            ax.set_xlabel('Joint pos')
            ax.set_ylabel('Constraint value')

    fig.savefig('plot/constraint/constraint_plot.png')

def contour_plot(comb, q, k_val, J=None, name=None, j_inv=True):   
    row = k_val[0].shape[-1] 
    col = len(comb)
    fig, axs = plt.subplots(row, col, figsize=(5*col, 5*row))
    
    for i in range(row):
        for j in range(col):
            ax = axs.flatten()[i * col + j] if row > 1 or col > 1 else axs
            contour(ax, q[j][:, 0], q[j][:, 1], k_val[j][:, i])
            if J is not None:
                if j_inv:
                    ax.quiver(q[j][:, 0], q[j][:, 1], J[j][:, 0, i], J[j][:, 1, i])
                else:
                    ax.quiver(q[j][:, 0], q[j][:, 1], J[j][:, i, 0], J[j][:, i, 1])
            ax.set_xlabel(f'Joint {comb[j][0]}')
            ax.set_ylabel(f'Joint {comb[j][1]}')

    fig.suptitle('Constraint value')
    fig.savefig(f'plot/constraint/{name + "_" if name else ""}{"J_inv_" if j_inv else "J_"}contour_plot.png')

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

