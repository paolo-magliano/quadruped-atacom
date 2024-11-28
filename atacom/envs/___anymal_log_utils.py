import torch


class AirHockeyConstrInfoLogger:
    def __init__(self, env_info):
        self.env_info = env_info
        self.n_episodes_total = 0
        self.n_episodes_joint = torch.zeros(self.env_info['robot']['n_joints'] * 2)
        self.n_episodes_link = torch.zeros(7)

        self.joint_pos_constr = -torch.inf * torch.ones(self.env_info['robot']['n_joints'] * 2)
        self.link_pos_constr = -torch.inf * torch.ones(7)
        self.not_count_joint = torch.tensor([True] * self.env_info['robot']['n_joints'] * 2)
        self.not_count_link = torch.tensor([True] * 7)
        self.puck_vel_cross = 0.

        # Compute Link Constraint Bound
        x_l = - self.env_info['robot']['base_frame'][0][0, 3] - (
            self.env_info['table']['length'] / 2 - self.env_info['mallet']['radius'])
        y_l = - (self.env_info['table']['width'] / 2 - self.env_info['mallet']['radius'])
        y_u = self.env_info['table']['width'] / 2 - self.env_info['mallet']['radius']
        z_l = self.env_info['robot']['ee_desired_height'] - 0.02
        z_u = self.env_info['robot']['ee_desired_height'] + 0.02
        z_wr = 0.35
        z_el = 0.35
        self.link_constr_ub = torch.tensor([x_l, y_l, -y_u, z_l, -z_u, z_wr, z_el])

    def update(self, joint_pos, link_pos, puck_obs):
        c_joint_pos = torch.cat([-joint_pos + self.env_info['robot']['joint_pos_limit'][0],
                                      joint_pos - self.env_info['robot']['joint_pos_limit'][1]])
        self.joint_pos_constr = c_joint_pos
        idx = torch.logical_and(c_joint_pos > 0, self.not_count_joint)
        self.n_episodes_joint[idx] += 1
        self.not_count_joint[idx] = False

        c_link_pos = link_pos + self.link_constr_ub
        self.link_pos_constr = c_link_pos
        idx = torch.logical_and(c_link_pos > 0, self.not_count_link)
        self.n_episodes_link[idx] += 1
        self.not_count_link[idx] = False

        if self.puck_vel_cross == 0. and puck_obs[0] > 0:
            self.puck_vel_cross = puck_obs[3]

    def episode_reset(self):
        self.n_episodes_total += 1
        self.joint_pos_constr = -torch.inf * torch.ones(self.env_info['robot']['n_joints'] * 2)
        self.link_pos_constr = -torch.inf * torch.ones(7)
        self.not_count_joint = torch.tensor([True] * self.env_info['robot']['n_joints'] * 2)
        self.not_count_link = torch.tensor([True] * 7)
        self.puck_vel_cross = 0

    def reset(self):
        self.n_episodes_joint = torch.zeros(self.env_info['robot']['n_joints'] * 2)
        self.n_episodes_link = torch.zeros(7)
        self.n_episodes_total = 0.

    def get_dict(self):
        return {'n_episodes': self.n_episodes_total,
                'n_episodes_link': self.n_episodes_link,
                'n_episodes_joint': self.n_episodes_joint,
                'joint_pos_constr': self.joint_pos_constr,
                'link_pos_constr': self.link_pos_constr,
                'puck_vel_cross': self.puck_vel_cross}


def get_dataset_info(dataset):
    start_idx = 0

    epoch_info = {}
    joint_pos = torch.tensor(len(dataset.info))
    link_pos = torch.tensor(len(dataset.info))
    success = torch.tensor(len(dataset.info))
    puck_vel_cross = torch.tensor(len(dataset.info))
    for i, info in enumerate(dataset.info):
        last = info[-1]
        if last:
            joint_pos = torch.stack(joint_pos, torch.max(dataset.info['joint_pos_constr'][start_idx: i+1], dim=0))
            link_pos = torch.stack(link_pos, torch.max(dataset.info['link_pos_constr'][start_idx: i+1], dim=0))
            success = torch.stack(success, dataset.info['success'][i])
            puck_vel_cross = torch.stack(puck_vel_cross, dataset.info['puck_vel_cross'][i])
            
            start_idx = i + 1


    epoch_info['joint_pos_constr'] = torch.mean(joint_pos, dim=0)
    epoch_info['link_pos_constr'] = torch.mean(link_pos, dim=0)
    epoch_info['n_episodes_joint'] = dataset.info['n_episodes_joint'][-1]
    epoch_info['n_episodes_link'] = dataset.info['n_episodes_link'][-1]
    epoch_info['success_rate'] = torch.sum(success) / len(success)
    epoch_info['puck_vel_cross'] = torch.mean(puck_vel_cross)
    epoch_info['n_episodes'] = dataset.info['n_episodes'][-1]
    return epoch_info
