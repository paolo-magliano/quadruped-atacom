import torch



class Observer:
    def __init__(self, num_envs, state_dim, dt, device, state_gain=50., distrurbance_gain=500.):
        self.num_envs = num_envs
        self.state_dim = state_dim
        self.dt = dt
        self.device = device
        self.state_gain = state_gain
        self.distrurbance_gain = distrurbance_gain

        # Initialize the state observer
        self.state_estimate = torch.zeros((num_envs, state_dim), device=device)
        self.state_dot_estimate = torch.zeros((num_envs, state_dim), device=device)
        self.disturbance_estimate = torch.zeros((num_envs, state_dim), device=device)
        self.disturbance_dot_estimate = torch.zeros((num_envs, state_dim), device=device)

    def update_estimate(self, state, input):
        # Update the state estimate using the observer equations
        self.state_dot_estimate = input + self.disturbance_estimate + self.state_gain * (state - self.state_estimate)
        self.disturbance_dot_estimate = self.distrurbance_gain * (state - self.state_estimate)
        self.state_estimate += self.state_dot_estimate * self.dt
        self.disturbance_estimate += self.disturbance_dot_estimate * self.dt

        return self.state_estimate, self.disturbance_estimate
    
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
