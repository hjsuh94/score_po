import numpy as np
import torch
import matplotlib.pyplot as plt

from score_po.score_matching import ScoreEstimator


class Environment:
    """
    Generate a 2D environment with obstacles.
    """

    def __init__(self):
        self.shape_lst = []
        
    def add_ellipse(self, mu, a, b):
        self.shape_lst.append(["ellipse", mu, a, b])
        
    def add_rectangle(self, mu, a, b):
        self.shape_lst.append(["rectangle", mu, a, b])
        
    def sample_ellipse(self, mu, a, b, samples):
        return ((samples[:,0] - mu[0]) / a) ** 2 + ((samples[:,1] - mu[1]) / b) ** 2 < 1.0

    def sample_rectangle(self, mu, a, b, samples):
        return torch.logical_and(
            (torch.abs(samples[:,0] - mu[0]) < a / 2),
            (torch.abs(samples[:,1] - mu[1]) < b / 2)
            )

    def sample_points(self, num_points):
        samples = 2.0 * torch.rand(num_points, 2) - 1.0
        valid = torch.zeros(num_points)
        for shape in self.shape_lst:
            if shape[0] == "ellipse":
                valid = torch.logical_or(valid,
                    self.sample_ellipse(shape[1], shape[2], shape[3], samples)
                )
            if shape[0] == "rectangle":
                valid = torch.logical_or(valid, 
                    self.sample_rectangle(shape[1], shape[2], shape[3], samples)
                )
        return samples[~valid,:]


def test():
    env = Environment()
    env.add_ellipse([0, 0], 0.4, 0.4)
    pts = env.sample_points(100000)
    plt.figure()
    plt.plot(pts[:,0], pts[:,1], 'ro')
    plt.savefig("env.png")
    plt.close()