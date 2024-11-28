import os
from tqdm import tqdm
import torch
import wandb
import inspect

from mushroom_rl.core import VectorCore, Logger
from mushroom_rl.utils.torch import TorchUtils

import hydra
from omegaconf import DictConfig
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict
from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *

from rl_util.build_rl_agent import build_rl_agent
from rl_util.compute_metric import compute_metrics

from util.wandb import wandb_init

from atacom.envs.anymal import AnymalEnv
from atacom_anymal import build_atacom_agent

@hydra.main(config_name="config", config_path="./cfg", version_base="1.1")
def experiment(cfg: DictConfig) -> None:
    cfg_dict = omegaconf_to_dict(cfg)

    TorchUtils.set_default_device(cfg_dict['rl_device'])
    torch.manual_seed(cfg_dict['seed'])

    logger = Logger(log_name=cfg_dict['task_name'], results_dir=cfg_dict['results_dir'], seed=cfg_dict['seed'], use_timestamp=True)
    wandb_run = wandb_init(cfg_dict)

    env, dynamics_info = AnymalEnv.build_env(cfg_dict, headless=cfg_dict['headless'])
    
    cfg_dict['atacom']['slack_beta'] = torch.tensor(cfg_dict['atacom']['slack_beta'])
    cfg_dict['atacom']['lambda_c'] = 0.5 / env.dt
    
    rl_agent = build_rl_agent(env.info, cfg_dict['train'])

    atacom_rl_agent = build_atacom_agent(rl_agent, dynamics_info, cfg_dict['atacom'])

    core = VectorCore(atacom_rl_agent, env)

    J, R, E, V = compute_metrics(core, cfg_dict['eval'])
    best_R = -float('inf')

    # Write logging
    logger.epoch_info(0, J=J, R=R, E=E, V=V)
    log_dict = {"Reward/J": J, "Reward/R": R, "Training/E": E, "Training/V": V}
    wandb.log(log_dict, step=0)
    
    if not cfg_dict['test']:
        for epoch in tqdm(range(cfg_dict['n_epochs']), disable=False, leave=False):           
            core.learn(**cfg_dict['learn'])

            J, R, E, V = compute_metrics(core, cfg_dict['eval'])

            # Write logging
            logger.epoch_info(epoch + 1, J=J, R=R, E=E, V=V)
            log_dict = {"Reward/J": J, "Reward/R": R, "Training/E": E, "Training/V": V}
            wandb.log(log_dict, step=epoch + 1)
            if R > best_R:
                best_R = R
                logger.log_best_agent(rl_agent, R)

            if (epoch + 1) % 100 == 0:
                logger.log_agent(rl_agent, epoch + 1)

        wandb_run.finish()

if __name__ == '__main__':
    experiment()
