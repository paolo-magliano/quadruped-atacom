
from omegaconf import OmegaConf
import os

import cProfile
import pstats
import argparse

import sys
sys.path.append('/home/magliano/Project/SafeLocomotion')

from experiments.util.hydra_cfg.reformat import omegaconf_to_dict
from experiments.util.hydra_cfg.hydra_utils import *

from experiments.util.plot_metric import load_dataset, plot_constraint, plot_metric, plot_experiment_metric
from atacom.envs.a1 import A1PIEnv
from experiments.atacom_a1 import constraint_list

def main() -> None:   
    parser = argparse.ArgumentParser(description='Plot metrics from A1 experiments.')
    parser.add_argument('--cfg_paths', type=str, nargs='+', default=None, help='Paths to the experiment folder containing the config.yaml file and dataset.')
    parser.add_argument('--compare_paths', type=str, nargs='+', default=['/home/stud_magliano/projects/SafeLocomotion/trained_policy/baseline/Velocity_baseline', '/home/stud_magliano/projects/SafeLocomotion/trained_policy/baseline/Position_baseline'], help='Paths to the baseline experiments for comparison.')
    parser.add_argument('--default_path', type=str, default='/home/stud_magliano/projects/SafeLocomotion/logs', help='Default path to the logs directory if no cfg_paths are provided.')
    parser.add_argument('--epsilon', type=float, default=0.,  help='Parameters to evaluate the constraints with epsilon margin from the border')
    args = parser.parse_args()
    if args.cfg_paths:
        cfg_paths = args.cfg_paths
    else:
        cfg_paths = [os.path.join(args.default_path, f) for f in os.listdir(args.default_path) if os.path.isdir(os.path.join(args.default_path, f)) and 'wandb' not in f]

    cfg = OmegaConf.load(os.path.join(cfg_paths[0], 'config.yaml'))
    cfg_dict = omegaconf_to_dict(cfg)
    cfg_dict['num_envs'] = 2
    env, env_info = A1PIEnv.build_env(cfg_dict)

    profile = cProfile.Profile()
    profile.enable()

    for cfg_path in cfg_paths:
        cfg_dict = omegaconf_to_dict(OmegaConf.load(os.path.join(cfg_path, 'config.yaml')))
        cfg_dict['num_envs'] = 2

        exp_paths = [cfg_path, *args.compare_paths] if args.compare_paths else [cfg_path]
        base_paths = [[os.path.join(exp_path, f) for f in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, f)) and 'plot' not in f and 'wandb' not in f] for exp_path in exp_paths]

        constr_list = constraint_list(cfg_dict['constraints'], env_info)

        plot_experiment_metric(base_paths, cfg_dict['n_epochs'], env_info['obs']['joint_pos_idx'], env_info['default_joint_pos'], constr_list=constr_list, epsilon=args.epsilon)
        print(f'Plotting done for {cfg_path}')

    profile.disable()
    stats = pstats.Stats(profile).sort_stats('cumtime')
    stats.print_stats(50)

    # path = '/home/stud_magliano/projects/SafeLocomotion/trained_policy/baseline/A1_PI_2025-04-02-09-30-42'
    # dataset_path = f'{path}/dataset'
    # plot_path = f'{path}/plot'plot_path
    # num_epoch = 15
    # epsilon = 0


    # env, env_info = A1PIEnv.build_env(cfg_dict)
    # dataset = load_dataset(dataset_path, num_epoch)
    
    # plot_constraint(dataset, num_epoch, plot_path, deep=True, epsilon=epsilon)
    # plot_constraint(dataset, num_epoch, plot_path, deep=False, epsilon=epsilon)

    # for i in range(num_epoch):
    #     plot_metric(dataset[i]['state'], env_info, i, plot_path)


        
if __name__ == '__main__':
    main()