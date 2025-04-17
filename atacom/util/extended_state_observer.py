import torch



class HighGainObserver:
    def __init__(self, num_envs, state_dim, dt, device, high_gain=1., state_gain=50., distrurbance_gain=500.):
        self.num_envs = num_envs
        self.state_dim = state_dim
        self.dt = dt
        self.device = device
        self.high_gain = high_gain
        self.state_gain = state_gain
        self.distrurbance_gain = distrurbance_gain

        # Initialize the state observer
        self.state_estimate = torch.zeros((num_envs, state_dim), device=device)
        self.state_dot_estimate = torch.zeros((num_envs, state_dim), device=device)
        self.disturbance_estimate = torch.zeros((num_envs, state_dim), device=device)
        self.disturbance_dot_estimate = torch.zeros((num_envs, state_dim), device=device)

    def update_estimate(self, state, input):
        # Update the state estimate using the observer equations
        self.state_dot_estimate = input + self.disturbance_estimate - self.high_gain * self.k_state(state)
        self.disturbance_dot_estimate = - self.high_gain ** 2 * self.k_disturbance(state)
        self.state_estimate += self.state_dot_estimate * self.dt
        self.disturbance_estimate += self.disturbance_dot_estimate * self.dt

        return self.state_estimate, self.disturbance_estimate
    
    def k_state(self, state):
        return self.state_gain * (self.state_estimate - state)
    
    def k_disturbance(self, state):
        return self.distrurbance_gain * (self.disturbance_estimate - state)
    
    def reset(self):
        # Reset the state observer
        self.state_estimate = torch.zeros_like(self.state_estimate)
        self.state_dot_estimate = torch.zeros_like(self.state_dot_estimate)
        self.disturbance_estimate = torch.zeros_like(self.disturbance_estimate)
        self.disturbance_dot_estimate = torch.zeros_like(self.disturbance_dot_estimate)

    def get_state_estimate(self):
        return self.state_estimate
    
    def get_disturbance_estimate(self):
        return self.disturbance_estimate

class MixedObserver(HighGainObserver):
    def k_state(self, state):
        error = self.state_gain * (self.state_estimate - state)
        return torch.sign(error) * torch.abs(error) ** 0.5 + error
    
    def k_disturbance(self, state):
        error = self.distrurbance_gain * (self.state_estimate - state)
        return torch.sign(error) * torch.abs(error) ** 0.5 + error + torch.sign((self.disturbance_estimate - state))