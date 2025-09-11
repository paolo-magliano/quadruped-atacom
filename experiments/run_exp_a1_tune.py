from tqdm import tqdm
import torch
import numpy
import wandb
import itertools

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
import pickle

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
    cfg_dict['test'] = True
    cfg_dict['num_envs'] = 256
    cfg_dict['horizon'] = 100
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
	# Kp: [1.25, 2.2, 1.1] Kd: [0.005, 0.01, 0.001] Ki: [0.15, 0.2, 0.55]
    base_kp = [1.25, 2.2, 1.1]
    base_kd = [0.005, 0.01, 0.001]
    base_ki = [0.15, 0.2, 0.55]
    Kp = [[1.2, 1.25, 1.3], [2.4, 2.45,  2.5, 2.55, 2.6, 2.7, 2.8], [1.15, 1.2, 1.25]]
    Kd = [[0.001, 0.005, 0.01, 0.05], [0.005, 0.01, 0.05, 0.1], [0., 0.001, 0.005]]
    Ki = [[0.15, 0.2, 0.25], [0.1, 0.15, 0.2, 0.25], [0.8, 0.85, 0.9, 0.95, 1., 1.1]]


    env._set_controller_gains(base_kp, base_kd, base_ki)

    env.stop(soft=False)
    env.controller_data.count=0
    env.controller_data.data = numpy.zeros((2, cfg_dict['num_envs'], env.controller_data.data_len, 12))
    J, R, E, V, task_info = compute_metrics(core, cfg_dict['eval'], env_info=env_info, deep_constr_log=cfg_dict['deep_constr_log'])
    best_R = -float('inf')
    error = numpy.square(env.controller_data.data[0, :, :100] - env.controller_data.data[1, :, :100]).mean()
    print(f'Base Error: {error}')

    joint_errors = [[], [], []]
    top = []
    for joint in [0, 1, 2]:
        joint_idx = [3*i + joint for i in range(4)]
        for kp in Kp[joint]:
            for kd in Kd[joint]:
                for ki in Ki[joint]:
                    test_kp = base_kp.copy()
                    test_kd = base_kd.copy()
                    test_ki = base_ki.copy()
                    test_kp[joint] = kp
                    test_kd[joint] = kd
                    test_ki[joint] = ki
                    env._set_controller_gains(test_kp, test_kd, test_ki)

                    env.stop(soft=False)
                    env.controller_data.count=0
                    env.controller_data.data = numpy.zeros((2, cfg_dict['num_envs'], env.controller_data.data_len, 12))
                    J, R, E, V, task_info = compute_metrics(core, cfg_dict['eval'], env_info=env_info, deep_constr_log=cfg_dict['deep_constr_log'])
                    best_R = -float('inf')
                    error = numpy.square(env.controller_data.data[0, :, :100, joint_idx] - env.controller_data.data[1, :, :100, joint_idx]).mean()
                    data = {
                        'error': error,
                        'Kp': kp,
                        'Kd': kd,
                        'Ki': ki
                    }
                    joint_errors[joint].append(data)
                    
        info = sorted(joint_errors[joint], key=lambda x: x["error"])
        print(f'Results for joint {joint}')
        top.append(info[:3])
        for i in info[:5]:# python experiments/run_exp_a1.py atacom.enable=False
            print(f'Error: {i["error"]}')
            print(f'\tKp: {i["Kp"]} Kd: {i["Kd"]} Ki: {i["Ki"]}')

    errors = []
    for data_0 in top[0]:
        for data_1 in top[1]:
            for data_2 in top[2]:
                test_kp = [data_0['Kp'], data_1['Kp'], data_2['Kp']]
                test_kd = [data_0['Kd'], data_1['Kd'], data_2['Kd']]
                test_ki = [data_0['Ki'], data_1['Ki'], data_2['Ki']]

                env._set_controller_gains(test_kp, test_kd, test_ki)

                env.stop(soft=False)
                env.controller_data.count=0
                env.controller_data.data = numpy.zeros((2, cfg_dict['num_envs'], env.controller_data.data_len, 12))
                J, R, E, V, task_info = compute_metrics(core, cfg_dict['eval'], env_info=env_info, deep_constr_log=cfg_dict['deep_constr_log'])
                best_R = -float('inf')
                error = numpy.square(env.controller_data.data[0, :, :100] - env.controller_data.data[1, :, :100]).mean()
                data = {
                    'error': error,
                    'Kp': test_kp,
                    'Kd': test_kd,
                    'Ki': test_ki
                }
                errors.append(data)

    info = sorted(errors, key=lambda x: x["error"])
    print(f'Combined Results')
    for i in info[:5]:
        print(f'Error: {i["error"]}')
        print(f'\tKp: {i["Kp"]} Kd: {i["Kd"]} Ki: {i["Ki"]}')

    with open('results.pkl', 'wb') as f:
        pickle.dump(info, f)

    profile = cProfile.Profile()
    profile.enable()
    if not cfg_dict['test']:
        for epoch in tqdm(range(cfg_dict['n_epochs']), disable=False, leave=False):
            core.learn(**cfg_dict['learn'])

            if cfg_dict['complete_eval']:
                J, R, E, V, task_info = compute_metrics(core, cfg_dict['eval'], cfg_dict['atacom']['enable'], env_info=env_info, deep_constr_log=cfg_dict['deep_constr_log'])

                # Write logging
                log_dict = log_info(logger, rl_agent, J, R, E, V, task_info, epoch)
                wandb.log(log_dict, step=epoch + 1)

                if R > best_R:
                    best_R = R
                    logger.log_best_agent(rl_agent, R)

                if (epoch + 1) % 10 == 0:
                    logger.log_agent(rl_agent, epoch + 1)

    profile.disable()
    stats = pstats.Stats(profile).sort_stats('cumtime')
    stats.print_stats(50)

if __name__ == '__main__':
    main()
