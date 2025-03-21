from tqdm import tqdm
import torch
import numpy
import wandb
from datetime import datetime

from mushroom_rl.core import VectorCore, Logger
from mushroom_rl.utils.torch import TorchUtils

import hydra
from omegaconf import DictConfig
from util.hydra_cfg.reformat import omegaconf_to_dict
from util.hydra_cfg.hydra_utils import *

from rl_util.build_rl_agent import build_rl_agent
from rl_util.compute_metric import compute_metrics
from rl_util.callbacks import LogDataset

from experiments.util.log_info import wandb_init, log_info, clean_dir

from atacom.envs.a1 import A1PDEnv, A1PIEnv
from atacom_a1 import build_atacom_agent

import cProfile
import pstats

@hydra.main(config_name="config", config_path="./cfg", version_base="1.1")
def main(cfg: DictConfig) -> None:

    cfg_dict = omegaconf_to_dict(cfg)

    TorchUtils.set_default_device('cuda')
    torch.manual_seed(cfg_dict['seed'])
    numpy.random.seed(cfg_dict['seed'])

    logger = Logger(log_name=cfg_dict['task_name'], results_dir=cfg_dict['results_dir'], seed=cfg_dict['seed'], use_timestamp=True)
    wandb_run = wandb_init(cfg_dict)

    try:
        experiment(cfg_dict, logger)
    finally:
        wandb_run.finish()
        clean_dir(logger._results_dir)

def experiment(cfg_dict, logger):
    if cfg_dict['control']['type'] == 'PI':
        env, env_info = A1PIEnv.build_env(cfg_dict)
    elif cfg_dict['control']['type'] == 'PD':
        env, env_info = A1PDEnv.build_env(cfg_dict)
    else:
        raise ValueError(f"Invalid control_type: {cfg_dict['control']['type']}")
    env.seed(cfg_dict['seed'])

    cfg_dict['atacom']['slack_beta'] = torch.tensor(cfg_dict['atacom']['slack_beta'])
    cfg_dict['atacom']['lambda_c'] = 2. / env.dt
    
    rl_agent = build_rl_agent(env.info, cfg_dict['train'])

    atacom_rl_agent = build_atacom_agent(rl_agent, env_info, cfg_dict['atacom'], cfg_dict['constraints'])

    callbacks_fit = []
    if not cfg_dict['complete_eval']:
        log_callback = LogDataset(atacom_rl_agent, cfg_dict['atacom']['enable'], env.info.gamma, logger, cfg_dict['eval']['n_episodes'])
        callbacks_fit.append(log_callback)

    record_params = {
        'path':logger._results_dir,
        'tag': 'records'
    }

    core = VectorCore(atacom_rl_agent, env, callbacks_fit=callbacks_fit, record_dictionary=record_params)

    if cfg_dict['complete_eval']:
        J, R, E, V, task_info = compute_metrics(core, cfg_dict['eval'], env_info=env_info, deep_constr_log=cfg_dict['deep_constr_log'], plot=cfg_dict['plot_actions'], epoch=-1, plot_path=logger._results_dir)
        best_R = -float('inf')

        # Write logging
        log_dict = log_info(logger, rl_agent, J, R, E, V, task_info, -1)
        wandb.log(log_dict, step=0)
        if cfg_dict['test'] and cfg_dict['record']:
            compute_metrics(core, cfg_dict['eval'], env_info=env_info)
            # wandb.log({"Policy": wandb.Video(f"{logger._results_dir}/records/recording-1.mp4", fps=(1 / env.dt))}, step=0)

    profile = cProfile.Profile()
    profile.enable()
    if not cfg_dict['test']:
        for epoch in tqdm(range(cfg_dict['n_epochs']), disable=False, leave=False):           
            core.learn(**cfg_dict['learn'])

            if cfg_dict['complete_eval']:
                J, R, E, V, task_info = compute_metrics(core, cfg_dict['eval'], env_info=env_info, deep_constr_log=cfg_dict['deep_constr_log'], plot=cfg_dict['plot_actions'], epoch=epoch, plot_path=logger._results_dir)

                # Write logging
                log_dict = log_info(logger, rl_agent, J, R, E, V, task_info, epoch)
                wandb.log(log_dict, step=epoch + 1)

                # if cfg_dict['record']:
                #         wandb.log({"Policy": wandb.Video(f"{logger._results_dir}/records/recording-{epoch + 2}.mp4", fps=(1 / env.dt))}, step=epoch + 1)
                        
                if R > best_R:
                    best_R = R
                    logger.log_best_agent(rl_agent, R)
                
                logger.log_agent(rl_agent, epoch + 1)
    
    profile.disable()
    stats = pstats.Stats(profile).sort_stats('cumtime')
    stats.print_stats(50)

if __name__ == '__main__':
    main()
