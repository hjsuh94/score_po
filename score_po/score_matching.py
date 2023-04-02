import numpy as np
import torch
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.data import TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

from score_po.nn import AdamOptimizerParams


class ScoreFunctionEstimator:
    """
    Score function estimator that stores the object
    ∇_z log p(z): R^(dim_x + dim_u) -> R^(dim_x + dim_u), where
    z = [x, u]^T. The class has functionalities for:
    1. Returning ∇_z log p(z), ∇_x log p(x,u), ∇_u log p(x,u)
    2. Training the estimator from existing data of (x,u) pairs.
    3. Training the
    """

    def __init__(self, network, dim_x, dim_u):
        self.net = network
        self.dim_x = dim_x
        self.dim_u = dim_u

    def get_score_z_given_z(self, z, sigma, eval=True):
        """
        input:
            z of shape (B, dim_x + dim_u)
            sigma, float
        output:
            ∇_z log p(z) of shape (B, dim_x + dim_u)
        """
        if eval:
            self.net.eval()

        input = torch.hstack((z, sigma * torch.ones(z.shape[0], 1)))
        return self.net(input)

    def get_score_x_given_z(self, z, sigma, eval=True):
        """Give ∇_x log p(z) part of the score function."""
        return self.get_score_z_given_z(z, sigma, eval)[:, : self.dim_x]

    def get_score_u_given_z(self, z, sigma, eval=True):
        """Give ∇_u log p(z) part of the score function."""
        return self.get_score_z_given_z(z, sigma, eval)[:, self.dim_x :]

    # The rest of the functions are same except they have x u as arguments.
    def get_score_z_given_xu(self, x, u, sigma, eval=True):
        z = torch.hstack((x, u))
        return self.get_score_z_given_z(z, sigma, eval)

    def get_score_x_given_xu(self, x, u, sigma, eval=True):
        z = torch.hstack((x, u))
        return self.get_score_x_given_z(z, sigma, eval)

    def get_score_x_given_xu(self, x, u, sigma, eval=True):
        z = torch.hstack((x, u))
        return self.get_score_u_given_z(z, sigma, eval)

    def evaluate_denoising_loss_with_sigma(self, data, sigma):
        """
        Evaluate denoising loss, Eq.(5) from Song & Ermon.
            data of shape (B, dim_x + dim_u)
            sigma, a scalar variable.
        """
        databar = data + torch.randn_like(data) * sigma
        target = -1 / (sigma**2) * (databar - data)
        scores = self.get_score_z_given_z(databar, sigma, eval=False)

        target = target.view(target.shape[0], -1)
        scores = scores.view(scores.shape[0], -1)

        loss = 0.5 * ((scores - target) ** 2).sum(dim=-1).mean(dim=0)
        return loss
    
    def evaluate_slicing_loss_with_sigma(self, data, sigma):
        """
        Evaluate denoising loss, Eq.(5) from Song & Ermon.
            data of shape (B, dim_x + dim_u)
            sigma, a scalar variable.
        """
        databar = data + torch.randn_like(data) * sigma
        databar.requires_grad_(True)
        
        vectors = torch.randn_like(databar)
        
        grad1 = self.get_score_z_given_z(databar, sigma)
        gradv = torch.sum(grad1 * vectors)
        grad2 = autograd.grad(gradv, databar, create_graph=True)[0]
        grad1 = grad1.view(databar.shape[0], -1)
        
        loss1 = torch.sum(grad1 * grad1, dim=-1) / 2.
        loss2 = torch.sum((vectors * grad2).view(databar.shape[0], -1), dim=-1)
        
        loss1 = loss1.view(1, -1).mean(dim=0)
        loss2 = loss2.view(1, -1).mean(dim=0)
        loss = loss1 + loss2

        return loss.mean()
    

    def evaluate_denoising_loss(self, data, sigma_lst):
        """
        Evaluate loss given input:
            data: of shape (B, dim_x + dim_u)
            sigma_lst: a geometric sequence of sigmas to train on.
        """
        loss = torch.zeros(1)
        for sigma in sigma_lst:
            loss += sigma**2.0 * self.evaluate_denoising_loss_with_sigma(data, sigma)
        return loss / len(sigma_lst)
    
    def evaluate_slicing_loss(self, data, sigma_lst):
        """
        Evaluate loss given input:
            data: of shape (B, dim_x + dim_u)
            sigma_lst: a geometric sequence of sigmas to train on.
        """
        loss = torch.zeros(1)
        for sigma in sigma_lst:
            loss += sigma**2.0 * self.evaluate_slicing_loss_with_sigma(data, sigma)
        return loss / len(sigma_lst)    

    def train_network(
        self,
        dataset: TensorDataset,
        params: AdamOptimizerParams,
        sigma_max=1,
        sigma_min=-3,
        n_sigmas=10,
    ):
        """
        Train a network given a dataset and optimization parameters.
        Following Song & Ermon, we train a noise-conditioned score function where
        the sequence of noise is provided with a geometric sequence of length
        n_sigmas, with max 10^log_sigma_max and min 10^log_sigma_min.
        """
        self.net.train()
        optimizer = optim.Adam(self.net.parameters(), params.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, params.iters)
        
        data_loader_train = torch.utils.data.DataLoader(
            dataset, batch_size=params.batch_size
        )
        data_loader_eval = torch.utils.data.DataLoader(
            dataset, batch_size=len(dataset)
        )

        sigma_lst = np.geomspace(sigma_min, sigma_max, n_sigmas)
        loss_lst = torch.zeros(params.iters)

        for epoch in tqdm(range(params.iters)):
            for z_batch in data_loader_train:
                z_batch = z_batch[0]
                optimizer.zero_grad()
                loss = self.evaluate_denoising_loss(z_batch, sigma_lst)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
            with torch.no_grad():
                for z_all in data_loader_eval:
                    z_all = z_all[0]
                    loss_eval = self.evaluate_denoising_loss(z_all, sigma_lst)
                    loss_lst[epoch] = loss_eval.item()
                print(f"epoch {epoch}, total loss {loss_eval.item()}")
                
        return loss_lst

    def save_network_parameters(self, filename):
        torch.save(self.net.state_dict(), filename)

    def load_network_parameters(self, filename):
        self.net.load_state_dict(torch.load(filename))
