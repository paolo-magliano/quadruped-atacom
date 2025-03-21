import wandb
import os
import sys
from math import floor, log10

def wandb_init(cfg_dict):
    mode = 'disabled' if cfg_dict['debug'] or cfg_dict['test'] or not cfg_dict['wandb']['activate'] else 'online'
    wandb.login(key=cfg_dict['wandb']['key'], timeout=10)
    return wandb.init(project=cfg_dict['wandb']['project'], dir=cfg_dict['results_dir'], config=cfg_dict, group=cfg_dict['wandb']['group'], mode=mode, entity=cfg_dict['wandb']['entity'])

def log_info(logger, rl_agent, J, R, E, V, task_info, epoch=None, precision=5, log_agent=False):
    log_dict = {"Reward/J": sig_figs(J, precision), "Reward/R": sig_figs(R, precision), "Training/E": sig_figs(E, precision), "Training/V": sig_figs(V, precision), "Episode_lenght": sig_figs(task_info['episode_length'], precision)}
    if epoch is not None:
        logger.epoch_info(epoch + 1, **{'J': sig_figs(J, precision), 'R': sig_figs(R, precision), 'E': sig_figs(E, precision), 'V': sig_figs(V, precision), 'Episode_lenght': sig_figs(task_info['episode_length'], precision)})
    else:
        logger.info(f'J: {sig_figs(J, precision)} R: {sig_figs(R, precision)} E: {sig_figs(E, precision)} V: {sig_figs(V, precision)} Episode_lenght: {sig_figs(task_info["episode_length"], precision)}')
    for key in ['constraints']:
        for name, value in task_info[key].items():
            costr_dict_wb = {}
            costr_dict = {}
            for k, v in value.items():
                costr_dict_wb[f"{key.capitalize()}/{name}/{k}"] = sig_figs(v, precision)
                costr_dict[f"{name}/{k}"] = sig_figs(v, precision)
            if epoch is not None:
                logger.epoch_info(epoch + 1, **costr_dict)
            else:
                logger.info(f'{name}: {costr_dict}')
            log_dict.update(costr_dict_wb)
    if len(task_info) > 1:
        logger.info('#' * 100)

    return log_dict

def sig_figs(x: float, precision: int):
    x = float(x)
    precision = int(precision)

    return round(x, -int(floor(log10(abs(x)))) + (precision - 1)) if x != 0 else 0

def clean_dir(dir_path):
    print(f'Cleaning directory {dir_path}: {os.path.exists(dir_path)} {os.path.isdir(dir_path)} {os.listdir(dir_path)}')
    if os.path.exists(dir_path) and os.path.isdir(dir_path) and not os.listdir(dir_path):
        os.rmdir(dir_path)

def clean_exit(sig, frame, dir_path):
    clean_dir(dir_path)
    sys.exit(0)