import torch.optim as optim
import torch.nn.functional as F

import os

from mushroom_rl.core import Agent
from mushroom_rl.rl_utils.preprocessors import MinMaxPreprocessor
from atacom.agent_builder.atacom_ppo import AtacomPPO
from mushroom_rl.policy.torch_policy import GaussianTorchPolicy

from .network import ActorNetwork, ActorNetwork


def _get_file_by_postfix(parent_dir, postfix):
    file_list = list()
    for root, _, files in os.walk(parent_dir):
        for f in files:
            if f.endswith(postfix):
                file_list.append(os.path.join(root, f))
    return file_list  

def build_rl_agent(mdp_info, cfg_dict):
    if not cfg_dict['checkpoint']:
        if cfg_dict['algorithm'] == 'PPO':
            return _build_agent_PPO(mdp_info, cfg_dict)
        else:
            raise NotImplementedError(f'Algorithm {cfg_dict["algorithm"]} not implemented')
    else:
        file_list = _get_file_by_postfix(cfg_dict['checkpoint'], f'{cfg_dict["seed"]}-best.msh')
        if file_list:
            return Agent.load(file_list[0])
        else:
            raise FileNotFoundError(f'No file cointaining {cfg_dict["seed"]}-best.msh found in {cfg_dict["checkpoint"]}')

def _build_agent_PPO(mdp_info, cfg_dict):
    cfg_dict['actor_opt']['class'] = optim.Adam
    
    cfg_dict['critic']['network'] = ActorNetwork
    cfg_dict['critic']['optimizer']['class'] = optim.Adam
    cfg_dict['critic']['loss'] = F.mse_loss
    cfg_dict['critic']['input_shape'] = mdp_info.observation_space.shape
    cfg_dict['critic']['output_shape'] = (1,)
    
    policy = GaussianTorchPolicy(ActorNetwork,
                                 mdp_info.observation_space.shape,
                                 mdp_info.action_space.shape,
                                 **cfg_dict['policy'])
    
    ppo_agent = AtacomPPO(mdp_info, policy, cfg_dict['actor_opt'], cfg_dict['critic'], **cfg_dict['params'])
    
    if cfg_dict['normalize_state']:
        ppo_agent.add_agent_preprocessor(MinMaxPreprocessor(mdp_info, 'torch'))

    return ppo_agent
