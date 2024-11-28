import wandb
import os

def wandb_init(cfg_dict):
    mode = 'disabled' if cfg_dict['debug'] or cfg_dict['test'] or not cfg_dict['wandb']['activate'] else 'online'
    key = os.getenv(cfg_dict['wandb']['key'])
    wandb.login(key=key, timeout=10)
    return wandb.init(project=cfg_dict['wandb']['project'], dir=cfg_dict['results_dir'], config=cfg_dict, group=cfg_dict['wandb']['group'], mode=mode, entity=cfg_dict['wandb']['entity'])

