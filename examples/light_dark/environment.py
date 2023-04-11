import numpy as np
import torch
import matplotlib.pyplot as plt

from score_po.dataset import Dataset
from score_po.score_matching import ScoreEstimator


class Environment:
    """
    Generate a 2D environment with obstacles.
    """

    def __init__(self):
        self.n_grid = 128
        self.grid = torch.ones((self.n_grid, self.n_grid))

    def generate_ellipse(self, a, b, mu):
        for i in range(self.n_grid):
            for j in range(self.n_grid):
                if ((i - mu[0]) / a) ** 2 + ((j - mu[1]) / b) ** 2 <= 1:
                    self.grid[i, j] = 0.0

    def generate_rectangle_2d(self, center, l_x, l_y):
        """
        Generate axis-aligned rectangle.
        points away from the obstacle region, and the boundary of the half plane
        passes through the point.
        TODO(terry-suh): change this to support rigid body transforms.
        """
        center_x = center[0]
        center_y = center[1]
        hl_x = int(l_x / 2)
        hl_y = int(l_y / 2)
        self.grid[
            center_x - hl_x : center_x + hl_x, center_y - hl_y : center_y + hl_y
        ] = 0.0

    def generate_star_2d(self, a, b, k):
        """
        Generate a star shape in 2d given by polar equations.
        """
        center = self.n_grid / 2
        for i in range(self.n_grid):
            for j in range(self.n_grid):
                nx = i - center
                ny = j - center
                r = torch.norm(torch.tensor([nx, ny]))
                theta = torch.atan2(torch.tensor(ny), torch.tensor(nx))
                if r < (a + b * torch.cos(k * theta)):
                    self.grid[i, j] = -1
        return self.grid

    def plot_environment(self):
        plt.imshow(self.grid.T, cmap="gray", origin="lower")

    def sample_points(self, num_points):
        samples = self.n_grid * torch.rand(num_points, 2)
        samples_int = torch.floor(samples).to(torch.long)
        sample_idx = self.grid[samples_int[:, 0], samples_int[:, 1]] == 1.0
        return samples[sample_idx, :]

    def sample_points_with_noise(self, num_points, sigma):
        samples = self.sample_points(num_points)
        return samples + torch.normal(0, sigma, samples.shape)


def plot_samples_and_enviroment(env, pts):
    plt.figure()
    env.plot_environment()
    plt.plot(pts[:, 0], pts[:, 1], "ro", markersize=1)
    plt.show()


def test():
    env = Environment()
    env.generate_ellipse(24, 24, [64, 64])

    pts = env.sample_points(1000)
    plot_samples_and_enviroment(env, pts)

    pts = env.sample_points_with_noise(5000, 3.0)
    plot_samples_and_enviroment(env, pts)
