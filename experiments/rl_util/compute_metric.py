import torch

def get_init_states(dataset):
    pick = True
    x_0 = list()
    for d in dataset:
        if pick:
            x_0.append(d[0])
        pick = d[-1]
    return torch.stack(x_0)


def compute_V(agent, dataset):
    Q = list()
    if hasattr(agent.learning_agent, "n_quantiles"):
        for state_orig in get_init_states(dataset):
            state = agent.learning_agent_preprocess(state_orig)
            s = torch.tensor([state for i in range(100)])
            a = torch.tensor([agent.draw_action(state_orig)[-agent.mdp_info.real_action_space.shape[0]:] for i in range(100)])
            tau = torch.linspace(0, 1, agent.learning_agent.n_quantiles + 1)
            tau = (tau[:-1] + tau[1:]) / 2
            tau = torch.tile(tau, (100, 1))
            Q_dist = agent.learning_agent._critic_approximator(s, a, tau)
            Q_mean = Q_dist.mean(axis=-1).mean()
            Q_std = Q_dist.std(axis=-1).mean()
            Q_median = Q_dist[:, 15].mean()
            Q_min = Q_dist[:, 0].mean()
            Q_max = Q_dist[:, -1].mean()
            Q.append([Q_mean, Q_std, Q_median, Q_min, Q_max])
    elif hasattr(agent.learning_agent, "_critic_approximator"):
        for state_orig in get_init_states(dataset):
            state = agent.learning_agent_preprocess(state_orig)
            s = torch.tensor([state for i in range(100)])
            a = torch.tensor([agent.draw_action(state_orig)[0][-agent.mdp_info.real_action_space.shape[0]:] for i in range(100)])
            Q.append(agent.learning_agent._critic_approximator(s, a).mean())
    elif hasattr(agent.learning_agent, "_V"):
        for state_orig in get_init_states(dataset):
            state = agent.learning_agent_preprocess(state_orig)
            Q.append(agent.learning_agent._V(state).mean())
    return torch.tensor(Q).mean(axis=0)


def compute_metrics(core, eval_params):
    if hasattr(core.env, "curriculum_training"):
        core.env.curriculum_training = False
    
    dataset = core.evaluate(**eval_params)

    rl_agent = core.agent.learning_agent
    J = torch.mean(dataset.compute_J(core.env.info.gamma))
    R = torch.mean(dataset.compute_J())

    if hasattr(rl_agent.policy, 'compute_action_and_log_prob'):
        _, log_prob = rl_agent.policy.compute_action_and_log_prob(
            core.agent.learning_agent_preprocess(dataset.parse()[0]))
        E = -log_prob.mean()
    else:
        E = rl_agent.policy.entropy(core.agent.learning_agent_preprocess(dataset.parse()[0]))

    Q_info = {}
    if hasattr(rl_agent, "n_quantiles"):
        Q_stats = compute_V(core.agent, dataset)
        Q_info = {"V_mean": Q_stats[0], "V_std": Q_stats[1],
                  "V_median": Q_stats[2], "V_min": Q_stats[3], "V_max": Q_stats[4]}
        V = Q_stats[0]
    else:
        V = compute_V(core.agent, dataset)

    if hasattr(core.env, 'clear_task_info'):
        core.env.clear_task_info()

    return J.item(), R.item(), E.item(), V.item()