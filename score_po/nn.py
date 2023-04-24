from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os, time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm

"""
List of architectures and parameters for NN training.
"""


@dataclass
class AdamOptimizerParams:
    lr: float = 1e-3
    epochs: int = 1000
    batch_size: int = 512


@dataclass
class WandbParams:
    enabled: bool = False
    project: Optional[str] = None
    name: Optional[str] = None
    dir: Optional[str] = None
    entity: Optional[str] = None
    config: Optional[Dict] = None

    def load_from_config(self, cfg: DictConfig, field: str):
        self.enabled = cfg[field].wandb.enabled
        self.project = cfg[field].wandb.project
        self.name = cfg[field].wandb.name
        self.dir = cfg[field].wandb.dir
        self.config = OmegaConf.to_container(cfg, resolve=True)
        self.entity = cfg[field].wandb.entity


@dataclass
class TrainParams:
    adam_params: AdamOptimizerParams
    wandb_params: WandbParams
    dataset_split: Tuple[float] = (0.9, 0.1)
    # Save the best model (with the smallest validation error to this path)
    save_best_model: Optional[str] = None
    # Device on which training occurs.
    device: str = "cuda"

    def __init__(self):
        self.adam_params = AdamOptimizerParams()
        self.wandb_params = WandbParams()

    def load_from_config(self, cfg: DictConfig):
        self.adam_params.batch_size = cfg.train.adam.batch_size
        self.adam_params.epochs = cfg.train.adam.epochs
        self.adam_params.lr = cfg.train.adam.lr
        self.wandb_params.load_from_config(cfg, "train")

        self.dataset_split = cfg.train.dataset_split
        self.save_best_model = cfg.train.save_best_model
        self.device = cfg.train.device


class MLP(nn.Module):
    """
    Vanilla MLP with ReLU nonlinearity.
    hidden_layers takes a list of hidden layers.
    For example,

    MLP(3, 5, [128, 128])

    makes MLP with two hidden layers with 128 width.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        hidden_layers: List[int],
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.hidden_layers = hidden_layers

        layers = []
        layers.append(nn.Linear(dim_in, hidden_layers[0]))
        layers.append(activation)
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(activation)
        layers.append(nn.Linear(hidden_layers[-1], dim_out))

        self.mlp = nn.Sequential(*layers)

    def get_vectorized_parameters(self) -> torch.Tensor:
        """
        Get a vectorized representation of the parameters.
        """
        params_vec = torch.zeros(0)
        for i in range(len(self.mlp)):
            if type(self.mlp[i]) == nn.Linear:
                params_vec = torch.hstack((params_vec, torch.ravel(self.mlp[i].weight)))
                params_vec = torch.hstack((params_vec, self.mlp[i].bias))
        return params_vec

    def set_vectorized_parameters(self, params_vector: torch.Tensor):
        """
        Get a vectorized representation of the parameters.
        """
        idx = 0
        for i in range(len(self.mlp)):
            layer = self.mlp[i]
            if type(layer) == nn.Linear:
                dim_in = layer.in_features
                dim_out = layer.out_features

                # Recover linear and bias parameters.
                linear_params = params_vector[idx : idx + dim_in * dim_out]
                idx += dim_in * dim_out
                bias_params = params_vector[idx : idx + dim_out]
                idx += dim_out

                # Set the NN parameters.
                layer.weight = nn.parameter.Parameter(
                    linear_params.view(dim_out, dim_in)
                )
                layer.bias = nn.parameter.Parameter(bias_params)

    def get_vectorized_gradients(self):
        """
        If the layers have registered gradients, extract them out to a vector
        representation.
        """
        device = self.mlp[0].bias.device
        params_vec = torch.zeros(0).to(device)
        for i in range(len(self.mlp)):
            if type(self.mlp[i]) == nn.Linear:
                params_vec = torch.hstack(
                    (params_vec, torch.ravel(self.mlp[i].weight.grad))
                )
                params_vec = torch.hstack((params_vec, self.mlp[i].bias.grad))
        return params_vec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class EnsembleNetwork(nn.Module):
    """
    Ensemble networks that contains a lists of identically sized NNs. Includes
    functionalities to compute the mean and variance among the ensembles.
    """

    def __init__(self, dim_in: torch.Size, dim_out: torch.Size, network_lst):
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.network_lst = network_lst
        self.K = len(network_lst)

    def forward(self, x_batch):
        """
        Get the mean of the ensemble by calling the ensemble. As a nn.Module,
        one could do

        ensemble = EnsembleNetwork(dim_in, dim_out, network_lst)
        y = ensemble(x)

        to get the mean.
        """
        B = x_batch.shape[0]
        batch = torch.zeros((self.K, B) + self.dim_out, device=x_batch.device)
        for k, network in enumerate(self.network_lst):
            batch[k] = network(x_batch)
        return batch.mean(dim=0)

    def get_einsum_string(self, length):
        """Get einsum string of specific length."""
        string = "ijklmnopqrstuvwxyz"
        if length > len(string):
            raise ValueError("dimension is larger than supported.")
        return string[:length]

    def get_variance(self, x_batch: torch.Tensor, metric: torch.Tensor = None):
        """
        Get the empirical variance of the ensemble given x_batch.
        metric is a torch.Tensor with the same shape as dim_out, and is used
        to evaluate the norm on the covariance matrix.

        If the metric is not provided, the evaluation will default to the two-norm.
        """

        if metric is None:
            metric = torch.ones(self.dim_out)
        metric = metric.to(x_batch.device)

        B = x_batch.shape[0]

        batch = torch.zeros((self.K, B) + self.dim_out, device=x_batch.device)
        for k, network in enumerate(self.network_lst):
            batch[k] = network(x_batch)
        mean = batch.mean(dim=0)
        dev = batch - mean.unsqueeze(0)  # K, B, dim_x
        # Note that empirical variance requires us to divide by K - 1 instead of K.
        e_str = self.get_einsum_string(len(self.dim_out))
        summation_string = "eb" + e_str + "," + e_str + "," + "eb" + e_str + "->eb"
        pairwise_dev = torch.einsum(summation_string, dev, metric, dev)

        return pairwise_dev.sum(dim=0) / (self.K - 1)

    def get_variance_gradients(self, x_batch, metric=None):
        x_batch = x_batch.clone()  # B x dim_x
        x_batch.requires_grad = True

        variance = self.get_variance(x_batch, metric)
        variance.sum().backward()
        return x_batch.grad

    def save_ensemble(self, foldername):
        if not os.path.exists(foldername):
            os.mkdir(foldername)
        for k, network in enumerate(self.network_lst):
            save_module(network, os.path.join(foldername, "{:02d}.pth".format(k)))

    def load_ensemble(self, foldername):
        for k, network in enumerate(self.network_lst):
            network.load_state_dict(os.path.join(foldername, "{:02d}.pth".format(k)))


def tuple_to_device(tup, device):
    lst = []
    for i in tup:
        lst.append(i.to(device))
    return tuple(lst)


def train_network(
    net: nn.Module, params: TrainParams, dataset: TensorDataset, loss_fn, split=True
):
    """
    Common utility function to train a neural network.
    net: nn.Module
    params: TrainParams
    loss_fn: a loss function for the optimization problem.
    loss_fn should have signature loss_fn(x_batch, net)
    """
    if params.wandb_params.enabled:
        if params.wandb_params.dir is not None and not os.path.exists(
            params.wandb_params.dir
        ):
            os.makedirs(params.wandb_params.dir, exist_ok=True)
        wandb.init(
            project=params.wandb_params.project,
            name=params.wandb_params.name,
            dir=params.wandb_params.dir,
            config=params.wandb_params.config,
            entity=params.wandb_params.entity,
        )

    net.train()

    optimizer = optim.Adam(net.parameters(), params.adam_params.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, params.adam_params.epochs
    )

    if split:
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, params.dataset_split
        )
    else:
        train_dataset = dataset
        val_dataset = dataset

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset, batch_size=params.adam_params.batch_size
    )
    data_loader_eval = torch.utils.data.DataLoader(
        val_dataset, batch_size=len(val_dataset)
    )

    loss_lst = np.zeros(params.adam_params.epochs)
    best_loss = np.inf

    training_loss = 0.0
    net = net.to(params.device)
    for epoch in tqdm(range(params.adam_params.epochs)):
        for z_batch in data_loader_train:
            optimizer.zero_grad()
            z_batch = tuple_to_device(z_batch, params.device)
            loss = loss_fn(z_batch, net)
            loss.backward()
            optimizer.step()
            scheduler.step()
            training_loss += loss.item() * z_batch[0].shape[0]
        training_loss /= len(train_dataset)

        with torch.no_grad():
            for z_all in data_loader_eval:
                z_all = tuple_to_device(z_all, params.device)
                loss_eval = loss_fn(z_batch, net)
                loss_lst[epoch] = loss_eval.item()
            if params.wandb_params.enabled:
                wandb.log(
                    {
                        "training_loss": training_loss,
                        "validation loss": loss_eval.item(),
                    },
                    step=epoch,
                )
            if params.save_best_model is not None and loss_eval.item() < best_loss:
                save_module(net, os.path.join(os.getcwd(), params.save_best_model))
                best_loss = loss_eval.item()

    return loss_lst


def train_network_sampling(
    net: nn.Module, params: TrainParams, sample_fn, loss_fn, split=True
):
    """
    A variant of train_network that does not use a dataset but a random sampling
    function. The sampling function should have the signature
    sample_fn(batch_size) and return (batch_size, dim_data).
    """
    if params.wandb_params.enabled:
        if params.wandb_params.dir is not None and not os.path.exists(
            params.wandb_params.dir
        ):
            os.mkdir(params.wandb_params.dir)
        wandb.init(
            project=params.wandb_params.project,
            name=params.wandb_params.name,
            dir=params.wandb_params.dir,
            config=params.wandb_params.config,
            entity=params.wandb_params.entity,
        )

    net.train()
    optimizer = optim.Adam(net.parameters(), params.adam_params.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, params.adam_params.epochs
    )

    loss_lst = np.zeros(params.adam_params.epochs)
    best_loss = np.inf

    for epoch in tqdm(range(params.adam_params.epochs)):
        optimizer.zero_grad()
        z_batch = sample_fn(params.adam_params.batch_size)
        loss = loss_fn(z_batch, net)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_eval = loss.clone().detach()
        loss_lst[epoch] = loss_eval.item()
        if params.wandb_params.enabled:
            wandb.log({"total loss": loss_eval.item()}, step=epoch)
        if params.save_best_model is not None and loss_eval.item() < best_loss:
            save_module(net, os.path.join(os.getcwd(), params.save_best_model))
            best_loss = loss_eval.item()

    return loss_lst


class Normalizer(nn.Module):
    """
    This class applies a shifting and scaling for the input. Namely if the input is x,
    then it outputs x̅(i) = (x(i) − b(i))/k(i), where b is the shifting term (bias),
    and k is the normalizing constant. We require this transformation to be invertible.

    Note that `b` and `k` are NOT parameters for optimization. They are constant values.
    `b` and `k` must have consistent size with x.
    """

    def __init__(self, k: torch.Tensor, b: torch.Tensor):
        """
        This module will outputx ̅(i) = (x(i) − b(i))/k(i)
        """
        super().__init__()
        self.register_buffer("k", k)
        self.register_buffer("b", b)

    def forward(self, x):
        return (x - self.b) / self.k

    def denormalize(self, xbar):
        return xbar * self.k + self.b


def save_module(nn_module: torch.nn.Module, filename: str):
    file_dir = os.path.dirname(filename)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir, exist_ok=True)
    torch.save(nn_module.state_dict(), filename)
