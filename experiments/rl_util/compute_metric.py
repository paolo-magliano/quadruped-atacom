import torch
from atacom.envs.costr_log_utils import get_dataset_info
import matplotlib.pyplot as plt

from experiments.kinematics_a1 import LinkPos

def get_init_states(dataset):
    pick = True
    x_0 = list()
    for d in dataset:
        if pick:
            x_0.append(d[0])
        pick = d[-1]
    return torch.stack(x_0)


def compute_V(agent, dataset, atacom_enable):
    Q = list()
    if atacom_enable:
        rl_agent = agent.learning_agent
    else:
        rl_agent = agent
    if hasattr(rl_agent, "n_quantiles"):
        for state_orig in get_init_states(dataset):
            state = agent.learning_agent_preprocess(state_orig) if atacom_enable else state_orig
            s = torch.tensor([state for i in range(100)])
            a = torch.tensor([agent.draw_action(state_orig)[-agent.mdp_info.real_action_space.shape[0]:] for i in range(100)])
            tau = torch.linspace(0, 1, rl_agent.n_quantiles + 1)
            tau = (tau[:-1] + tau[1:]) / 2
            tau = torch.tile(tau, (100, 1))
            Q_dist = rl_agent._critic_approximator(s, a, tau)
            Q_mean = Q_dist.mean(axis=-1).mean()
            Q_std = Q_dist.std(axis=-1).mean()
            Q_median = Q_dist[:, 15].mean()
            Q_min = Q_dist[:, 0].mean()
            Q_max = Q_dist[:, -1].mean()
            Q.append([Q_mean, Q_std, Q_median, Q_min, Q_max])
    elif hasattr(rl_agent, "_critic_approximator"):
        for state_orig in get_init_states(dataset):
            state = agent.learning_agent_preprocess(state_orig) if atacom_enable else state_orig
            s = torch.tensor([state for i in range(100)])
            a = torch.tensor([agent.draw_action(state_orig)[0][-agent.mdp_info.real_action_space.shape[0]:] for i in range(100)])
            Q.append(agent.learning_agent._critic_approximator(s, a).mean())
    elif hasattr(rl_agent, "_V"):
        for state_orig in get_init_states(dataset):
            state = agent.learning_agent_preprocess(state_orig) if atacom_enable else state_orig
            Q.append(rl_agent._V(state).mean())
    return torch.tensor(Q).mean(axis=0)


def compute_metrics(core, eval_params, atacom_enable, deep_constr_log=False, plot=False, env_info=None, epoch=None, plot_path=None):
    if hasattr(core.env, "curriculum_training"):
        core.env.curriculum_training = False
    
    dataset = core.evaluate(**eval_params)

    if plot:
        plot_hist(dataset.state.cpu(), env_info, epoch)

    J, R, E, V, task_info = get_metrics(dataset, core.agent, atacom_enable, core.env.info.gamma, deep_constr_log)

    if hasattr(core.env, 'clear_task_info'):
        core.env.clear_task_info()

    return J, R, E, V, task_info

def plot_hist(state, env_info, epoch=None, plot_path=None):
    feet_names = ['FL', 'FR', 'RL', 'RR']
    joint_names = ['Hip', 'Thigh', 'Calf']
    joint_pos = state[:, [ 6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]]

    # Plot histogram of joint angles positions
    fig, axs = plt.subplots(4, 3, figsize=(15, 20))
    for i in range(4):
        for j in range(3):
            axs[i, j].hist(joint_pos[:, i * 3 + j], bins=100)
            axs[i, j].set_title(f"{feet_names[i]} {joint_names[j]}")

    plt.savefig(f"{plot_path if plot_path is not None else '.'}/plot/distribution/joint_pos_distribution_{epoch if epoch is not None else ''}.png")

    # Plot histogram of foot positions
    if env_info is not None:
        feet = [LinkPos(env_info['urdf_path'], side + '_foot', side + '_thigh', env_info['default_joint_pos'],  env_info['action']['idx'][side]) for side in feet_names]

        fig, axs = plt.subplots(4, 3, figsize=(15, 20))
        for i in range(4):
            for j in range(3):
                foot = feet[i]
                foot_pos = foot.get_pos(joint_pos.cuda()).cpu()
                axs[i, j].hist(foot_pos[:, j], bins=100)
                axs[i, j].set_title(f"{feet_names[i]} {['x', 'y', 'z'][j]}")

        plt.savefig(f"{plot_path if plot_path is not None else '.'}/plot/distribution/feet_pos_distribution_{epoch if epoch is not None else ''}.png")

def get_metrics(dataset, agent, atacom_enable, gamma, deep_constr_log=False):
    J = torch.mean(dataset.compute_J(gamma))
    R = torch.mean(dataset.compute_J())

    if atacom_enable:
        rl_agent = agent.learning_agent
    else:
        rl_agent = agent

    entropy_states = agent.learning_agent_preprocess(dataset.parse()[0]) if atacom_enable else dataset.parse()[0]
    entropy_states = entropy_states[:5000]
    if hasattr(rl_agent.policy, 'compute_action_and_log_prob'):
        _, log_prob = rl_agent.policy.compute_action_and_log_prob(entropy_states)
        E = -log_prob.mean()
    else:
        E = rl_agent.policy.entropy(entropy_states)

    Q_info = {}
    if hasattr(rl_agent, "n_quantiles"):
        Q_stats = compute_V(agent, dataset, atacom_enable)
        Q_info = {"V_mean": Q_stats[0], "V_std": Q_stats[1],
                  "V_median": Q_stats[2], "V_min": Q_stats[3], "V_max": Q_stats[4]}
        V = Q_stats[0]
    else:
        V = compute_V(agent, dataset, atacom_enable)

    task_info = get_dataset_info(dataset, deep_constr_log)
    task_info['episode_length'] = torch.mean(dataset.episodes_length.float()).item()
    task_info.update(Q_info)

    return J.item(), R.item(), E.item(), V.item(), task_info

    
