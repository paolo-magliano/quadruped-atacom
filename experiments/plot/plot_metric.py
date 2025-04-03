import hydra
from omegaconf import DictConfig

import sys
sys.path.append('/home/magliano/Project/SafeLocomotion')

from experiments.util.hydra_cfg.reformat import omegaconf_to_dict
from experiments.util.hydra_cfg.hydra_utils import *

from experiments.util.plot_metric import load_dataset, plot_constraint, plot_metric
from atacom.envs.a1 import A1PDEnv, A1PIEnv

@hydra.main(config_name="config", config_path="../cfg", version_base="1.1")
def main(cfg: DictConfig) -> None:
    cfg_dict = omegaconf_to_dict(cfg)

    path = '/home/magliano/Project/SafeLocomotion/trained_policy/baseline/A1_PI_2025-04-02-09-30-42'
    dataset_path = f'{path}/dataset'
    plot_path = f'{path}/plot'
    num_epoch = 15
    epsilon = 0


    env, env_info = A1PIEnv.build_env(cfg_dict)
    dataset = load_dataset(dataset_path, num_epoch)
    
    plot_constraint(dataset, num_epoch, plot_path, deep=True, epsilon=epsilon)
    plot_constraint(dataset, num_epoch, plot_path, deep=False, epsilon=epsilon)

    for i in range(num_epoch):
        plot_metric(dataset[i]['state'], env_info, i, plot_path)
        
if __name__ == '__main__':
    main()