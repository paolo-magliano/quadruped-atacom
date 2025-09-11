from tqdm import tqdm
import torch
import numpy
import wandb
import pickle

from mushroom_rl.core import VectorCore, Logger
from mushroom_rl.utils.torch import TorchUtils

import hydra
from omegaconf import DictConfig
from  experiments.util.hydra_cfg.reformat import omegaconf_to_dict
from  experiments.util.hydra_cfg.hydra_utils import *

from experiments.rl_util.build_rl_agent import build_rl_agent
from experiments.rl_util.compute_metric import compute_metrics
from experiments.rl_util.callbacks import LogDataset

from experiments.util.log_info import wandb_init, log_info, clean_dir

from atacom.envs.a1 import A1PDEnv, A1PIEnv
from experiments.atacom_a1 import build_atacom_agent

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
    cfg_dict['num_envs'] = 2
    cfg_dict['eval']['n_episodes'] = 2
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

    if cfg_dict['atacom']['enable']:
        atacom_rl_agent = build_atacom_agent(rl_agent, env_info, cfg_dict['atacom'], cfg_dict['constraints'])
    else:
        atacom_rl_agent = rl_agent

    callbacks_fit = []
    if not cfg_dict['complete_eval']:
        log_callback = LogDataset(atacom_rl_agent, cfg_dict['atacom']['enable'], env.info.gamma, logger, cfg_dict['eval']['n_episodes'])
        callbacks_fit.append(log_callback)

    core = VectorCore(atacom_rl_agent, env, callbacks_fit=callbacks_fit)

    Kp = [0.9, 0.95, 1, 1.05, 1.1]
    Kd = [0., 0.001, 0.005]
    Ki = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
    for kp in Kp:
        env._Kp = kp
        for kd in Kd:
            env._Kd = kd
            for ki in Ki:
                env._Ki = ki
                J, R, E, V, task_info = compute_metrics(core, cfg_dict['eval'], cfg_dict['atacom']['enable'], env_info=env_info, deep_constr_log=cfg_dict['deep_constr_log'])

                # Write logging
                log_dict = log_info(logger, rl_agent, J, R, E, V, task_info, -1)
                wandb.log(log_dict, step=0)

    info = sorted(env.controller_tune, key=lambda x: x["error"])
    # Save info
    with open('info.pkl', 'wb') as f:
        pickle.dump(info, f)


    for i in info[:20]:
        print(f'Error: {i["error"]}')
        print(f'\tKp: {i["Kp"]} Kd: {i["Kd"]} Ki: {i["Ki"]}')
        

if __name__ == '__main__':
    main()

