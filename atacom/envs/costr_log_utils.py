import torch
import matplotlib.pyplot as plt

class ConstrLogger:
    def __init__(self, num_envs):
        self.data = [{} for _ in range(num_envs)]

    def log(self, name, value):
        for i in range(len(self.data)):
            self.data[i][name] = value[i].clone().detach()
    
    def get_and_reset(self):
        data = self.data
        self.data = [{} for _ in range(len(data))]
        return data
    
    def empty(self):
        return all([len(d) == 0 for d in self.data])

def get_dataset_info(dataset, deep=False):
    info = { 'constraints': {} }
    for key in dataset.info.keys():
        if deep:
            for i in range(dataset.info[key].shape[1]):
                info['constraints'][f'{key}_{i}'] = _metrics_dict(dataset.info[key][:, i])
        else:
            info['constraints'][key] = _metrics_dict(dataset.info[key])

    return info

def _metrics_dict(data):
    bool_violation = data > 0
    return {
        'violation_mean': data[bool_violation].mean().item() if bool_violation.any() else 0.,
        # 'violation_std': data[bool_violation].std().item() if bool_violation.any() else 0.,
        # 'num_violation': bool_violation.any(dim=-1).sum().item(),
        'average_violation': bool_violation.any(dim=-1).float().mean().item()
    }
