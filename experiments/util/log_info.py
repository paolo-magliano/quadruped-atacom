import wandb
import os
from math import floor, log10

def wandb_init(cfg_dict):
    mode = 'disabled' if cfg_dict['debug'] or cfg_dict['test'] or not cfg_dict['wandb']['activate'] else 'online'
    key = os.getenv(cfg_dict['wandb']['key'])
    wandb.login(key=key, timeout=10)
    return wandb.init(project=cfg_dict['wandb']['project'], dir=cfg_dict['results_dir'], config=cfg_dict, group=cfg_dict['wandb']['group'], mode=mode, entity=cfg_dict['wandb']['entity'])

def log_info(logger, rl_agent, epoch, J, R, E, V, task_info, precision=5):
    log_dict = {"Reward/J": sig_figs(J, precision), "Reward/R": sig_figs(R, precision), "Training/E": sig_figs(E, precision), "Training/V": sig_figs(V, precision), "Episode_lenght": sig_figs(task_info['episode_length'], precision)}
    logger.epoch_info(epoch + 1, **{'J': sig_figs(J, precision), 'R': sig_figs(R, precision), 'E': sig_figs(E, precision), 'V': sig_figs(V, precision), 'Episode_lenght': sig_figs(task_info['episode_length'], precision)})
    for name, value in task_info['constraints'].items():
        costr_dict_wb = {}
        costr_dict = {}
        for k, v in value.items():
            costr_dict_wb[f"Constraint/{name}/{k}"] = sig_figs(v, precision)
            costr_dict[f"{name}/{k}"] = sig_figs(v, precision)
        if value['num_violation'] > 0 and epoch > 0:
            logger.log_agent(rl_agent, str(epoch + 1) + '_' + name)
        logger.epoch_info(epoch + 1, **costr_dict)
        log_dict.update(costr_dict_wb)
    if len(task_info) > 1:
        logger.info('#' * 100)

    return log_dict

def sig_figs(x: float, precision: int):
    x = float(x)
    precision = int(precision)

    return round(x, -int(floor(log10(abs(x)))) + (precision - 1)) if x != 0 else 0