from score_po.dynamical_system import DynamicalSystem
import torch
import torch.nn as nn
import numpy as np

class SingleIntegratorPixels(DynamicalSystem):
    def __init__(self):
        super().__init__(2, 2)
        self.is_differentiable = True

    def dynamics(self, x, u):
        return x + u

    def dynamics_batch(self, x_batch, u_batch):
        return x_batch + u_batch

class SingleIntegratorPixelsAverage(DynamicalSystem):
    def __init__(self):
        super().__init__(2, 2)
        self.is_differentiable = True
        self.y_mesh, self.x_mesh = torch.meshgrid(torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 32))
        self.ds_lst = None
        # self.x_mesh = self.x_mesh.T
        self.y_mesh_u, self.x_mesh_u = torch.meshgrid(torch.linspace(-0.3, 0.3, 32), torch.linspace(-0.3, 0.3, 32))

    def dynamics(self, x_batch, u_batch, eps=1e-6):
        x_batch = x_batch.reshape(1, 2)
        u_batch = u_batch.reshape(1, 32, 32)
        u_norm = u_batch.clamp(min=0.0, max=1.) / (u_batch.clamp(min=0.0, max=1.) + eps).sum(dim=(1, 2))[:, None, None].repeat(
            1, u_batch.shape[-2], u_batch.shape[-1])
        pos_x = (u_norm * self.x_mesh_u.to(u_norm.device)).sum(dim=(1, 2))
        pos_y = (u_norm * self.y_mesh_u.to(u_norm.device)).sum(dim=(1, 2))
        u_batch = torch.hstack([pos_x[:, None], pos_y[:, None]])
        return (x_batch + u_batch).flatten()

    def dynamics_batch(self, x_batch, u_batch, eps=1e-6):
        x_batch = x_batch.reshape(x_batch.shape[0], 2)
        u_batch = u_batch.reshape(u_batch.shape[0], 32, 32)
        u_batch = u_batch.reshape(u_batch.shape[0], int(np.sqrt(u_batch.shape[1])), int(np.sqrt(u_batch.shape[1])))
        u_norm = u_batch.clamp(min=0.0, max=1.) / (u_batch.clamp(min=0.0, max=1.) + eps).sum(dim=(1, 2))[:, None, None].repeat(
            1, u_batch.shape[-2], u_batch.shape[-1])
        pos_x = (u_norm * self.x_mesh_u.to(u_norm.device)).sum(dim=(1, 2))
        pos_y = (u_norm * self.y_mesh_u.to(u_norm.device)).sum(dim=(1, 2))
        u_batch = torch.hstack([pos_x[:, None], pos_y[:, None]])
        return x_batch + u_batch

dynamics = SingleIntegratorPixels()
