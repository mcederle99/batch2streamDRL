import numpy as np
import gymnasium as gym
import torch

# code from ...
class SampleMeanStd:
    def __init__(self, shape=(), device="cpu"):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.p = np.ones(shape, "float64")
        self.count = 0
        self.device = device

    def update(self, x):
        if self.count == 0:
            self.mean = x
            self.p = np.zeros_like(x)
        self.mean, self.var, self.p, self.count = self.update_mean_var_count_from_moments(self.mean, self.p, self.count, x*1.0)

    def update_mean_var_count_from_moments(self, mean, p, count, sample):
        new_count = count + 1
        new_mean = mean + (sample - mean) / new_count
        p = p + (sample - mean) * (sample - new_mean)
        new_var = 1 if new_count < 2 else p / (new_count - 1)
        return new_mean, new_var, p, new_count
    
    def normalize(self, x):
         if isinstance(x, np.ndarray):
             x = torch.Tensor(x).to(self.device)
         return (x - torch.Tensor(self.mean).to(self.device)) / torch.sqrt(torch.Tensor(self.var).to(self.device) + 1e-8)
    

class SampleStd:
    def __init__(self, shape=(), device="cpu"):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.p = np.ones(shape, "float64")
        self.count = 0
        self.device = device

    def update(self, x):
        if self.count == 0:
            self.mean = x
            self.p = np.zeros_like(x)
        self.mean, self.var, self.p, self.count = self.update_mean_var_count_from_moments(self.mean, self.p, self.count, x*1.0)

    def update_mean_var_count_from_moments(self, mean, p, count, sample):
        new_count = count + 1
        new_mean = mean + (sample - mean) / new_count
        p = p + (sample - mean) * (sample - new_mean)
        new_var = 1 if new_count < 2 else p / (new_count - 1)
        return new_mean, new_var, p, new_count
    
    def normalize(self, x):
         if x.shape[0] == 1 and isinstance(x, np.ndarray):
             x = torch.Tensor(x).to(self.device)
         return x / torch.sqrt(torch.Tensor(self.var).to(self.device) + 1e-8)