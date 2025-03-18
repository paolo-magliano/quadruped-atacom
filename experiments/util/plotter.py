import matplotlib.pyplot as plt
import numpy as np
import torch  
import os

class Plotter():
    def __init__(self, data_dim, data_len=500, n_row=1, n_col=1, title='plot', path='plot', data_labels=None):
        self.data = np.empty((data_dim, data_len, n_row * n_col))
        self.data_len = data_len
        self.n_row = n_row
        self.n_col = n_col
        self.title = title
        self.path = path
        self.data_labels = data_labels
        self.count = 0
        self.n_plot = 0

    def add_data(self, *data):
        data = self._conver_data(*data)
        assert data.shape[0] == self.data.shape[0]
        assert data.shape[1] == self.data.shape[2]
        self.data[:, self.count, :] = data
        self.count += 1
        if self.count == self.data_len:
            self.plot()
            self.count = 0

    def _conver_data(self, *data):
        data = [d.cpu().numpy() if isinstance(d, torch.Tensor) else d for d in data]
        return np.vstack(data)
    
    def plot(self):
        fig, axes = plt.subplots(self.n_row, self.n_col, figsize=(15, 20))
        for i in range(self.n_row):
            for j in range(self.n_col):
                idx = i * self.n_col + j
                for k in range(self.data.shape[0]):
                    axes[i, j].plot(self.data[k, :, idx], label=self.data_labels[k] if self.data_labels is not None else None, alpha=0.7)
                axes[i, j].legend()
                axes[i, j].set_title(f'{self.title} {idx}')

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        plt.savefig(f'{self.path}/{self.title}_{self.n_plot}.png')
        self.n_plot += 1



