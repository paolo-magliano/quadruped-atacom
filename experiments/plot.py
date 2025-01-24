import isaacsim
import matplotlib.pyplot as plt
from atacom.envs.a1 import A1Env
import torch
import math
from mushroom_rl.utils.isaac_sim import ObservationType
import tqdm

def main():
    cfg = {
        'num_envs': 1,
        'horizon': 1000,
        'headless': False,
        'control': {
            'type': 'velocity',
            'p_gains': 0.7, # 0.7
            'd_gains': 2e-4, # 2e-4
            'action_scale': 5. # 5.
        },
        'urdf_filepath': 'atacom/envs/assets/a1.urdf'
    }
    vel_scale = 0.05
    env = A1Env(cfg)

    obs = env.reset()

    acts = []
    vel = []

    joint = 2
    for i in tqdm.tqdm(range(200)):
        action = torch.full((cfg['num_envs'], env.info.action_space.shape[0]), 0., device=torch.device('cuda'))
        action [:, joint] = math.sin(i / 2)
        obs, reward, done, info = env.step(action)
        acts.append(obs[0, env.observation_helper.obs_idx_map['actions']][joint].item() * cfg['control']['action_scale'])
        vel.append(obs[0, env.observation_helper.obs_types_idx_map[ObservationType.JOINT_VEL]][joint].item() / vel_scale)

    fig, ax = plt.subplots()
    ax.plot(acts[100:])
    ax.plot(vel[100:])
    ax.legend(['action', 'velocity'])
    plt.savefig('plot.png')

if __name__ == '__main__':
    main()
        
    