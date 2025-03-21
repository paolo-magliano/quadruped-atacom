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

def get_dataset_info(dataset, deep=False):
    info = { 
        'constraints': get_constraint_info(dataset, deep=deep)
    }

    return info

def get_constraint_info(dataset, deep=False):
    info = {}
    for key in dataset.info.keys():
        if 'constraint' in key:
            if deep:
                for i in range(dataset.info[key].shape[1]):
                    info[f'{key}_{i}'] = _constr_metrics_dict(dataset.info[key][:, i])
            else:
                info[key] = _constr_metrics_dict(dataset.info[key])

    return info

def _constr_metrics_dict(data):
    bool_violation = data > 0
    return {
        'violation_mean': data[bool_violation].mean().item() if bool_violation.any() else 0.,
        # 'violation_std': data[bool_violation].std().item() if bool_violation.any() else 0.,
        # 'num_violation': bool_violation.any(dim=-1).sum().item(),
        'average_violation': bool_violation.any(dim=-1).float().mean().item()
    }
