import torch
import matplotlib.pyplot as plt

class ConstrLogger:
    def __init__(self):
        self.data = {}

    def log(self, name, value):
        self.data[f'{name}_constraint'] = value.clone().detach()
    
    def get_and_reset(self):
        data = self.data
        self.data = {}
        return data
    
    def empty(self):
        return len(self.data) == 0

def get_dataset_info(dataset, deep=False, full=False):
    info = { 
        'constraints': get_constraint_info(dataset, deep=deep, full=full),
    }

    return info

def get_constraint_info(dataset, deep=False, full=False, epsilon=0):
    info = {}
    for key in dataset.info.keys():
        if 'constraint' in key:
            info[key] = _constr_metrics_dict(dataset.info[key], full=full, epsilon=epsilon)
            if deep:
                for i in range(dataset.info[key].shape[1]):
                    info[f'{key}_{i}'] = _constr_metrics_dict(dataset.info[key][:, i], full=full, epsilon=epsilon)

    return info

def _constr_metrics_dict(data, full=False, epsilon=0):
    bool_violation = data > epsilon
    metrics = {
        'violation_mean': data[bool_violation].mean().item() if bool_violation.any() else 0.,
        # 'violation_std': data[bool_violation].std().item() if bool_violation.any() else 0.,
        # 'num_violation': bool_violation.any(dim=-1).sum().item(),
        'average_violation': bool_violation.any(dim=-1).float().mean().item() if bool_violation.ndim > 1 else bool_violation.float().mean().item()
    }
    if full:
        metrics['violation_max'] = data[bool_violation].max().item() if bool_violation.any() else 0.
        metrics['violation_min'] = data[bool_violation].min().item() if bool_violation.any() else 0.
        metrics['violation_median'] = data[bool_violation].median().item() if bool_violation.any() else 0.
        metrics['violation_std'] = data[bool_violation].std().item() if bool_violation.any() else 0.

    return metrics

def plot_joint_constr(joint_pos, joint_space_low, joint_space_high):
    n_point = 100
    middle = (joint_space_high + joint_space_low) / 2
    joint_limit = torch.cat([torch.linspace(0, joint_space_high[i] - middle[i], n_point).unsqueeze(-1) for i in range(middle.shape[0])], dim=-1)
    x = joint_limit / (joint_space_high - middle) * 100
    y = torch.zeros(n_point, joint_pos.shape[1])
    for i in range(n_point):
        violation_high = joint_pos > (middle + joint_limit[i])
        violation_low = joint_pos < (middle - joint_limit[i])
        violation = torch.logical_or(violation_high, violation_low).float().mean(dim=0) * 100
        y[i] = violation

    return x, y

if __name__ == '__main__':
    feet_names = ['FL', 'FR', 'RL', 'RR']
    joint_names = ['Hip', 'Thigh', 'Calf']
    # Env joint pos limit: tensor([[-0.8029, -1.0472, -2.6965, -0.8029, -1.0472, -2.6965, -0.8029, -1.0472,
    #      -2.6965, -0.8029, -1.0472, -2.6965],
    #     [ 0.8029,  4.1888, -0.9163,  0.8029,  4.1888, -0.9163,  0.8029,  4.1888,
    #      -0.9163,  0.8029,  4.1888, -0.9163]])
    default = torch.tensor([
            0.1, 0.8, -1.5,
            -0.1, 0.8, -1.5,
            0.1, 1., -1.5,
            -0.1, 1., -1.5
        ])
    joint_space_low = torch.tensor([-0.8029, -1.0472, -2.6965, -0.8029, -1.0472, -2.6965, -0.8029, -1.0472, -2.6965, -0.8029, -1.0472, -2.6965])
    joint_space_high = torch.tensor([0.8029, 4.1888, -0.9163, 0.8029, 4.1888, -0.9163, 0.8029, 4.1888, -0.9163, 0.8029, 4.1888, -0.9163])
    fig, ax = plt.subplots(4, 3, figsize=(15, 20))
    ax = ax.flatten()
    epoch = [1, 2, 3, 5, 10, 15]
    x = torch.zeros(len(epoch), 100, 12)
    y = torch.zeros(len(epoch), 100, 12)
    for idx, i in enumerate(epoch):
        state = torch.load(f'/home/stud_magliano/projects/SafeLocomotion/logs/A1_2025-03-25-14-33-36/dataset/state_{i}.pt')
        joint_pos = state[:, [ 6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]]

        joint_pos = joint_pos + default

        xx, yy = plot_joint_constr(joint_pos, joint_space_low, joint_space_high)
        x[idx] = xx
        y[idx] = yy

    fig, ax = plt.subplots(4, 3, figsize=(15, 20))
    ax = ax.flatten()
    for i in range(12):
        for j in range(len(epoch)):
            ax[i].plot(x[j, :, i], y[j, :, i], label=f'Epoch {epoch[j]}')
        ax[i].set_title(f'{feet_names[i % 4]} {joint_names[i // 4]}')
        ax[i].set_xlabel('Joint range (%)')
        ax[i].set_ylabel('Violation (%)')
        ax[i].legend()

    plt.savefig('plot/joint_constr_training.png')

