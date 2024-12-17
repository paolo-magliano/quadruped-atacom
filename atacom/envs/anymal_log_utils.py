import torch

class AnymalConstrLogger:
    def __init__(self, num_envs):
        self.data = [{} for _ in range(num_envs)]

    def log(self, name, value):
        for i in range(len(self.data)):
            self.data[i][name] = value[i]
    
    def get_and_reset(self):
        data = self.data
        self.data = [{} for _ in range(len(data))]
        return data
    
    def empty(self):
        return all([len(d) == 0 for d in self.data])

def get_dataset_info(dataset):
    info = { 'constraints': {} }
    for key in dataset.info.keys():
        bool_violation = dataset.info[key] > 0
        info['constraints'][key] = {
                'violation_mean': dataset.info[key][bool_violation].mean().item() if bool_violation.any() else 0.,
                # 'violation_std': dataset.info[key][bool_violation].std().item() if bool_violation.any() else 0.,
                'num_violation': bool_violation.any(dim=1).sum().item(),
                'average_violation': bool_violation.any(dim=1).float().mean().item()
        }

    return info