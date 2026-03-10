from tqdm import tqdm
import torch
import numpy
import wandb
from datetime import datetime
import os

from mushroom_rl.core import VectorCore, Logger
from mushroom_rl.utils.torch import TorchUtils

import hydra
from omegaconf import DictConfig, OmegaConf
from util.hydra_cfg.reformat import omegaconf_to_dict
from util.hydra_cfg.hydra_utils import *

from rl_util.build_rl_agent import build_rl_agent
from rl_util.compute_metric import compute_metrics
from rl_util.callbacks import LogDataset

from util.log_info import wandb_init, log_info, clean_dir
from util.plot_metric import plot_experiment_metric

from atacom.envs.a1 import A1PIDEnv, A1Pos, A1Vel
from atacom_a1 import build_atacom_agent

import cProfile
import pstats

@hydra.main(config_name="config", config_path="./cfg", version_base="1.1")
def main(cfg: DictConfig) -> None:

    profile = cProfile.Profile()
    profile.enable()

    cfg_dict = omegaconf_to_dict(cfg)

    TorchUtils.set_default_device('cuda')
    exp_dir = 'exp_' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') if cfg_dict['experiment_subdir'] is None else cfg_dict['experiment_subdir']
    cfg_dict['results_dir'] = os.path.join(cfg_dict['results_dir'], exp_dir)
    wandb_run = wandb_init(cfg_dict)
    os.makedirs(cfg_dict['results_dir'], exist_ok=True)
    OmegaConf.save(config=cfg, f=f'{cfg_dict["results_dir"]}/config.yaml')
    
    base_paths = []

    if cfg_dict['control']['type'] == 'Pos':
        env, env_info = A1Pos.build_env(cfg_dict)
    elif cfg_dict['control']['type'] == 'Vel':
        env, env_info = A1Vel.build_env(cfg_dict)
    else:
        env, env_info = A1PIDEnv.build_env(cfg_dict)

    cfg_dict['atacom']['slack_beta'] = torch.tensor(cfg_dict['atacom']['slack_beta'])
    cfg_dict['atacom']['lambda_c'] = cfg_dict['atacom']['lambda_c'] / env.dt
    cfg_dict['atacom']['lambda_integral'] = cfg_dict['atacom']['lambda_integral'] / env.dt

    for seed in cfg_dict['seed']:
        torch.manual_seed(seed)
        numpy.random.seed(seed)

        rl_agent = build_rl_agent(env.info, cfg_dict['train'])

        atacom_rl_agent = build_atacom_agent(rl_agent, env_info, cfg_dict['atacom'], cfg_dict['constraints'])

        logger = Logger(log_name=f'{cfg_dict["task_name"]}_{seed}', results_dir=cfg_dict['results_dir'], seed=seed, use_timestamp=True)
        base_paths.append(logger._results_dir)

        try:
            experiment(cfg_dict, env, env_info, atacom_rl_agent, logger, seed)
        finally:
            clean_dir(logger._results_dir)
    if not cfg_dict['test']:
        compare_paths = [[os.path.join(exp_path, f) for f in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, f)) and 'plot' not in os.path.join(exp_path, f)] for exp_path in cfg_dict['plot']['compare_paths']]
        plot_experiment_metric([base_paths, *compare_paths], cfg_dict['n_epochs'], env_info['obs']['joint_pos_idx'], env_info['default_joint_pos'], constr_list=atacom_rl_agent.atacom_controller.constraints)
    
    wandb_run.finish()

    profile.disable()
    stats = pstats.Stats(profile).sort_stats('cumtime')
    stats.print_stats(50)

def experiment(cfg_dict, env, env_info, atacom_rl_agent, logger, seed):
    env.seed(seed)

    record_params = {
        'path':logger._results_dir,
        'tag': 'records'
    }

    core = VectorCore(atacom_rl_agent, env, record_dictionary=record_params)
        
    J, R, E, V, task_info = compute_metrics(core, cfg_dict['eval'], env_info, 0, deep_constr_log=cfg_dict['deep_constr_log'], base_path=logger._results_dir)
    best_R = -float('inf')

    # Write logging
    log_dict = log_info(logger, atacom_rl_agent.learning_agent, J, R, E, V, task_info, -1)
    wandb.log(log_dict, step=0)

    if not cfg_dict['test']:
        for epoch in tqdm(range(cfg_dict['n_epochs']), disable=False, leave=False):           
            core.learn(**cfg_dict['learn'])

            if (epoch + 2) == cfg_dict['n_epochs']:
                cfg_dict['eval']['render'] = True
                cfg_dict['eval']['record'] = True

            J, R, E, V, task_info = compute_metrics(core, cfg_dict['eval'], env_info, epoch + 1, deep_constr_log=cfg_dict['deep_constr_log'], base_path=logger._results_dir)

            # Write logging
            log_dict = log_info(logger, atacom_rl_agent.learning_agent, J, R, E, V, task_info, epoch)
            wandb.log(log_dict, step=epoch + 1)
                    
            logger.log_best_agent(atacom_rl_agent.learning_agent, R)

    env.stop(soft=False)
    atacom_rl_agent.learning_agent.stop()

    if cfg_dict['record'] and os.path.exists(f"{logger._results_dir}/records/recording-{cfg_dict['n_epochs']}.mp4"):
        wandb.log({"Policy": wandb.Video(f"{logger._results_dir}/records/recording-{cfg_dict['n_epochs']}.mp4", fps=(1 / env.dt))})

if __name__ == '__main__':
    main()
