
from omegaconf import OmegaConf
import os

import argparse

from util.hydra_cfg.reformat import omegaconf_to_dict
from util.hydra_cfg.hydra_utils import *

from util.plot_metric import load_dataset, plot_constraint, plot_metric, plot_experiment_metric
from atacom.envs.a1 import A1PIDEnv
from atacom_a1 import constraint_list

def main() -> None:   
    parser = argparse.ArgumentParser(description='Plot metrics from A1 experiments')
    parser.add_argument('--cfg_paths', type=str, nargs='+', default=None, help='Paths to the experiment folder containing the config.yaml file and dataset.')
    parser.add_argument('--compare_paths', type=str, nargs='+', default=None, help='Paths to the baseline experiments for comparison.')
    parser.add_argument('--default_path', type=str, default='./logs', help='Default path to the logs directory if no cfg_paths are provided.')
    parser.add_argument('--epsilon', type=float, default=0.,  help='Parameters to evaluate the constraints with epsilon margin from the border')
    args = parser.parse_args()
    if args.cfg_paths:
        cfg_paths = args.cfg_paths
    else:
        cfg_paths = [os.path.join(args.default_path, f) for f in os.listdir(args.default_path) if os.path.isdir(os.path.join(args.default_path, f)) and 'wandb' not in f]

    cfg = OmegaConf.load(os.path.join(cfg_paths[0], 'config.yaml'))
    cfg_dict = omegaconf_to_dict(cfg)
    cfg_dict['num_envs'] = 2
    env, env_info = A1PIDEnv.build_env(cfg_dict)

    for cfg_path in cfg_paths:
        cfg_dict = omegaconf_to_dict(OmegaConf.load(os.path.join(cfg_path, 'config.yaml')))
        cfg_dict['num_envs'] = 2

        exp_paths = [cfg_path, *args.compare_paths] if args.compare_paths else [cfg_path]
        base_paths = [[os.path.join(exp_path, f) for f in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, f)) and 'plot' not in f and 'wandb' not in f] for exp_path in exp_paths]

        constr_list = constraint_list(cfg_dict['constraints'], env_info)

        plot_experiment_metric(base_paths, cfg_dict['n_epochs'], env_info['obs']['joint_pos_idx'], env_info['default_joint_pos'], constr_list=constr_list, epsilon=args.epsilon)
        print(f'Plotting done for {cfg_path}')

        
if __name__ == '__main__':
    main()