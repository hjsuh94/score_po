from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os, time
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
from functools import partial
import torch.nn.functional as F

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
        activation: nn.Module = nn.ELU(),
        layer_norm: bool = False
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
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_layers[i + 1]))
            layers.append(activation)
        layers.append(nn.Linear(hidden_layers[-1], dim_out))

        self.mlp = nn.Sequential(*layers)

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
        self.params, self.buffers = torch.func.stack_module_state(
            network_lst)
        
        base_model = copy.deepcopy(network_lst[0])
        base_model = base_model.to('meta')
        
        def fmodel(params, buffers, x):
            return torch.func.functional_call(base_model, (
                params, buffers), (x,))
            
        self.map = torch.vmap(fmodel)

    def forward(self, x_batch):
        """ Return the model in batch. """
        # https://pytorch.org/tutorials/intermediate/ensembling.html
        if x_batch.shape[0] != self.K:
            raise ValueError("leading dimension must be equal to ensemble size.")
        return self.map(self.params, self.buffers, x_batch)

    def get_mean(self, x_batch):
        B = x_batch.shape[0]
        batch = torch.zeros(self.K, B, self.dim_out, device=x_batch.device)
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

        batch = torch.zeros(self.K, B, self.dim_out, device=x_batch.device)
        for k, network in enumerate(self.network_lst):
            batch[k] = network(x_batch)
        mean = batch.mean(dim=0)
        dev = batch - mean.unsqueeze(0)  # K, B, dim_x
        # Note that empirical variance requires us to divide by K - 1 instead of K.
        e_str = self.get_einsum_string(1)
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
    net: nn.Module,
    params: TrainParams,
    dataset: TensorDataset,
    loss_fn,
    split=True,
    callback=None
):
    """
    Common utility function to train a neural network.
    net: nn.Module
    params: TrainParams
    loss_fn: a loss function for the optimization problem.
    loss_fn should have signature loss_fn(x_batch, net)
    callback: should have signature callback(net, loss, epoch)
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
    net = net.to(params.device)    

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
        
        if callback is not None:
            callback(net, loss.item(), epoch)
            
        with torch.no_grad():
            for z_all in data_loader_eval:
                z_all = tuple_to_device(z_all, params.device)
                loss_eval = loss_fn(z_batch, net)
                loss_lst[epoch] = loss_eval.item()
            if params.wandb_params.enabled:
                wandb.log(
                    {
                        "training_loss": training_loss,
                        "validation3 loss": loss_eval.item(),
                    },
                    step=epoch,
                )
            if params.save_best_model is not None and loss_eval.item() < best_loss:
                save_module(net, os.path.join(os.getcwd(), params.save_best_model))
                best_loss = loss_eval.item()

    return loss_lst


def train_network_sampling(
    net: nn.Module, params: TrainParams, sample_fn, loss_fn,
    callback):
    """
    A variant of train_network that does not use a dataset but a random sampling
    function. The sampling function should have the signature
    sample_fn(batch_size) and return (batch_size, dim_data).
    callback: should have signature callback(net, loss, epoch)
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
        
        if callback is not None:
            callback(net, loss.item(), epoch)        

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

def tensor_linspace(start, end, steps=10):
    # linspace in torch, adopted from https://github.com/zhaobozb/layout2im.
    """
    Vectorized version of torch.linspace.
    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer
    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
      out.select(-1, 0) == start, out.select(-1, -1) == end,
      and the other elements of out linearly interpolate between
      start and end.
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out.to(start.device)


class MLPwEmbedding(torch.nn.Module):
    """
    Train an MLP with Embedding.
    hidden_layers takes a list of hidden layers, 
    and S takes a positive integer value that corresponds to number of tokens.
    For example,

    MLP(dim_in=3, dim_out=5, S=10, nn_layers=[128, 128])

    creates a MLP with two hidden layers with 128 width.
    
    The architecture choice the MLPwEmbedding assumes that all the 
    nn layers are of equal size.
    """
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        hidden_layers: List[int],
        embedding_size: int, # number of tokens
        activation: nn.Module = nn.ELU(),
    ):
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.embedding_size = embedding_size
        self.hidden_layers = hidden_layers
        self.act = activation
        
        # assert all elements of hideen layers are equal.
        if len(set(hidden_layers)) != 1:
            raise ValueError("hidden_layers must have same elements.")

        self.initial = nn.Linear(dim_in, hidden_layers[0])
        self.embed = nn.Embedding(embedding_size, hidden_layers[0])
        self.final = nn.Linear(hidden_layers[-1], dim_out)

        self.layers = []
        for i in range(len(hidden_layers)-1):
            self.layers.append(LinearBlock(hidden_layers[i], hidden_layers[i+1]))
        self.hl = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor, s: torch.int) -> torch.Tensor:
        s = torch.Tensor([s]).type(torch.int).to(x.device)
        x = self.initial(x)
        x = self.act(x)
        for i in range(len(self.hidden_layers) - 1):
            x = self.layers[i](x, self.embed(s))
        x = self.final(x)
        return x


class LinearBlock(torch.nn.Module):
    def __init__(self, dim_in, dim_out, activation=nn.ELU()):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.act = activation
        self.layernorm = nn.LayerNorm(dim_out)
        
    def forward(self, x, y):
        x = self.linear(x) * y
        x = self.layernorm(x)
        x = self.act(x)
        return x

def generate_cosine_schedule(sigma_max, sigma_min, steps):
    # normalize space between 0 to 1.
    x = 0.5 * (torch.cos(torch.linspace(0, torch.pi, steps)) + 1)
    return (sigma_max - sigma_min) * x + sigma_min

############################################ Image-based networks ############################################
def get_normalization(conditional=True):
    if conditional:
        return ConditionalInstanceNorm2dPlus
    else:
        return InstanceNorm2dPlus

def get_sigmas(config):
    sigmas = torch.tensor(
        np.exp(np.linspace(np.log(config.model.sigma_begin), np.log(config.model.sigma_end),
                           config.model.num_classes))).float().to(config.train.device)
    return sigmas

def get_act():
    return nn.ELU()

def spectral_norm(layer, n_iters=1):
    return torch.nn.utils.spectral_norm(layer, n_power_iterations=n_iters)

def dilated_conv3x3(in_planes, out_planes, dilation, bias=True, spec_norm=False):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=dilation, dilation=dilation, bias=bias)
    if spec_norm:
        conv = spectral_norm(conv)
    return conv

def conv1x1(in_planes, out_planes, stride=1, bias=True, spec_norm=False):
    "1x1 convolution"
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=bias)
    if spec_norm:
        conv = spectral_norm(conv)
    return conv

def conv3x3(in_planes, out_planes, stride=1, bias=True, spec_norm=False):
    "3x3 convolution with padding"
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)
    if spec_norm:
        conv = spectral_norm(conv)
    return conv

class InstanceNorm2dPlus(nn.Module):
    def __init__(self, num_features, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)
        self.alpha = nn.Parameter(torch.zeros(num_features))
        self.gamma = nn.Parameter(torch.zeros(num_features))
        self.alpha.data.normal_(1, 0.02)
        self.gamma.data.normal_(1, 0.02)
        if bias:
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        means = torch.mean(x, dim=(2, 3))
        m = torch.mean(means, dim=-1, keepdim=True)
        v = torch.var(means, dim=-1, keepdim=True)
        means = (means - m) / (torch.sqrt(v + 1e-5))
        h = self.instance_norm(x)

        if self.bias:
            h = h + means[..., None, None] * self.alpha[..., None, None]
            out = self.gamma.view(-1, self.num_features, 1, 1) * h + self.beta.view(-1, self.num_features, 1, 1)
        else:
            h = h + means[..., None, None] * self.alpha[..., None, None]
            out = self.gamma.view(-1, self.num_features, 1, 1) * h
        return out
class ConditionalInstanceNorm2dPlus(nn.Module):
    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)
        if bias:
            self.embed = nn.Embedding(num_classes, num_features * 3)
            self.embed.weight.data[:, :2 * num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
            self.embed.weight.data[:, 2 * num_features:].zero_()  # Initialise bias at 0
        else:
            self.embed = nn.Embedding(num_classes, 2 * num_features)
            self.embed.weight.data.normal_(1, 0.02)

    def forward(self, x, y):
        means = torch.mean(x, dim=(2, 3))
        m = torch.mean(means, dim=-1, keepdim=True)
        v = torch.var(means, dim=-1, keepdim=True)
        means = (means - m) / (torch.sqrt(v + 1e-5))
        h = self.instance_norm(x)

        if self.bias:
            gamma, alpha, beta = self.embed(y).chunk(3, dim=-1)
            h = h + means[..., None, None] * alpha[..., None, None]
            out = gamma.view(-1, self.num_features, 1, 1) * h + beta.view(-1, self.num_features, 1, 1)
        else:
            gamma, alpha = self.embed(y).chunk(2, dim=-1)
            h = h + means[..., None, None] * alpha[..., None, None]
            out = gamma.view(-1, self.num_features, 1, 1) * h
        return out
class ConvMeanPool(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True, adjust_padding=False, spec_norm=False):
        super().__init__()
        if not adjust_padding:
            conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
            if spec_norm:
                conv = spectral_norm(conv)
            self.conv = conv
        else:
            conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
            if spec_norm:
                conv = spectral_norm(conv)

            self.conv = nn.Sequential(
                nn.ZeroPad2d((1, 0, 1, 0)),
                conv
            )

    def forward(self, inputs):
        output = self.conv(inputs)
        output = sum([output[:, :, ::2, ::2], output[:, :, 1::2, ::2],
                      output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
        return output
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, resample=None, act=nn.ELU(),
                 normalization=nn.BatchNorm2d, adjust_padding=False, dilation=None, spec_norm=False):
        super().__init__()
        self.non_linearity = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample
        self.normalization = normalization
        if resample == 'down':
            if dilation is not None:
                self.conv1 = dilated_conv3x3(input_dim, input_dim, dilation=dilation, spec_norm=spec_norm)
                self.normalize2 = normalization(input_dim)
                self.conv2 = dilated_conv3x3(input_dim, output_dim, dilation=dilation, spec_norm=spec_norm)
                conv_shortcut = partial(dilated_conv3x3, dilation=dilation, spec_norm=spec_norm)
            else:
                self.conv1 = conv3x3(input_dim, input_dim, spec_norm=spec_norm)
                self.normalize2 = normalization(input_dim)
                self.conv2 = ConvMeanPool(input_dim, output_dim, 3, adjust_padding=adjust_padding,
                                          spec_norm=spec_norm)
                conv_shortcut = partial(ConvMeanPool, kernel_size=1, adjust_padding=adjust_padding,
                                        spec_norm=spec_norm)

        elif resample is None:
            if dilation is not None:
                conv_shortcut = partial(dilated_conv3x3, dilation=dilation, spec_norm=spec_norm)
                self.conv1 = dilated_conv3x3(input_dim, output_dim, dilation=dilation, spec_norm=spec_norm)
                self.normalize2 = normalization(output_dim)
                self.conv2 = dilated_conv3x3(output_dim, output_dim, dilation=dilation, spec_norm=spec_norm)
            else:
                # conv_shortcut = nn.Conv2d ### Something wierd here.
                conv_shortcut = partial(conv1x1, spec_norm=spec_norm)
                self.conv1 = conv3x3(input_dim, output_dim, spec_norm=spec_norm)
                self.normalize2 = normalization(output_dim)
                self.conv2 = conv3x3(output_dim, output_dim, spec_norm=spec_norm)
        else:
            raise Exception('invalid resample value')

        if output_dim != input_dim or resample is not None:
            self.shortcut = conv_shortcut(input_dim, output_dim)

        self.normalize1 = normalization(input_dim)

    def forward(self, x):
        output = self.normalize1(x)
        output = self.non_linearity(output)
        output = self.conv1(output)
        output = self.normalize2(output)
        output = self.non_linearity(output)
        output = self.conv2(output)

        if self.output_dim == self.input_dim and self.resample is None:
            shortcut = x
        else:
            shortcut = self.shortcut(x)

        return shortcut + output

class RCUBlock(nn.Module):
    def __init__(self, features, n_blocks, n_stages, act=nn.ReLU(), spec_norm=False):
        super().__init__()

        for i in range(n_blocks):
            for j in range(n_stages):
                setattr(self, '{}_{}_conv'.format(i + 1, j + 1), conv3x3(features, features, stride=1, bias=False,
                                                                         spec_norm=spec_norm))

        self.stride = 1
        self.n_blocks = n_blocks
        self.n_stages = n_stages
        self.act = act

    def forward(self, x):
        for i in range(self.n_blocks):
            residual = x
            for j in range(self.n_stages):
                x = self.act(x)
                x = getattr(self, '{}_{}_conv'.format(i + 1, j + 1))(x)

            x += residual
        return x
class CRPBlock(nn.Module):
    def __init__(self, features, n_stages, act=nn.ReLU(), maxpool=True, spec_norm=False):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(n_stages):
            self.convs.append(conv3x3(features, features, stride=1, bias=False, spec_norm=spec_norm))
        self.n_stages = n_stages
        if maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        else:
            self.maxpool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)

        self.act = act

    def forward(self, x):
        x = self.act(x)
        path = x
        for i in range(self.n_stages):
            path = self.maxpool(path)
            path = self.convs[i](path)
            x = path + x
        return x
class MSFBlock(nn.Module):
    def __init__(self, in_planes, features, spec_norm=False):
        """
        :param in_planes: tuples of input planes
        """
        super().__init__()
        assert isinstance(in_planes, list) or isinstance(in_planes, tuple)
        self.convs = nn.ModuleList()
        self.features = features

        for i in range(len(in_planes)):
            self.convs.append(conv3x3(in_planes[i], features, stride=1, bias=True, spec_norm=spec_norm))

    def forward(self, xs, shape):
        sums = torch.zeros(xs[0].shape[0], self.features, *shape, device=xs[0].device)
        for i in range(len(self.convs)):
            h = self.convs[i](xs[i])
            h = F.interpolate(h, size=shape, mode='bilinear', align_corners=True)
            sums += h
        return sums
class RefineBlock(nn.Module):
    def __init__(self, in_planes, features, act=nn.ReLU(), start=False, end=False, maxpool=True, spec_norm=False):
        super().__init__()

        assert isinstance(in_planes, tuple) or isinstance(in_planes, list)
        self.n_blocks = n_blocks = len(in_planes)

        self.adapt_convs = nn.ModuleList()
        for i in range(n_blocks):
            self.adapt_convs.append(
                RCUBlock(in_planes[i], 2, 2, act, spec_norm=spec_norm)
            )

        self.output_convs = RCUBlock(features, 3 if end else 1, 2, act, spec_norm=spec_norm)

        if not start:
            self.msf = MSFBlock(in_planes, features, spec_norm=spec_norm)

        self.crp = CRPBlock(features, 2, act, maxpool=maxpool, spec_norm=spec_norm)

    def forward(self, xs, output_shape):
        assert isinstance(xs, tuple) or isinstance(xs, list)
        hs = []
        for i in range(len(xs)):
            h = self.adapt_convs[i](xs[i])
            hs.append(h)

        if self.n_blocks > 1:
            h = self.msf(hs, output_shape)
        else:
            h = hs[0]

        h = self.crp(h)
        h = self.output_convs(h)

        return h
class NCSNv2(nn.Module):
    def __init__(self, config, input_dim, output_dim, layers):
        super().__init__()
        self.logit_transform = False
        self.rescaled = False
        self.norm = get_normalization(conditional=False)
        self.ngf = ngf = config.model.ngf
        self.num_classes = num_classes = config.model.num_classes

        self.act = act = get_act()
        self.register_buffer('sigmas', get_sigmas(config))
        self.config = config

        # self.dim_in = dim_in
        # self.dim_out = dim_out

        self.begin_conv = nn.Conv2d(config.data.channels, ngf, 3, stride=1, padding=1)

        self.normalizer = self.norm(ngf, self.num_classes)
        self.end_conv = nn.Conv2d(ngf, config.data.channels, 3, stride=1, padding=1)

        self.res1 = nn.ModuleList([
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                          normalization=self.norm),
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res2 = nn.ModuleList([
            ResidualBlock(self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res3 = nn.ModuleList([
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm, dilation=2),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm, dilation=2)]
        )

        self.res4 = nn.ModuleList([
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm, adjust_padding=False, dilation=4),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm, dilation=4)]
        )

        self.refine1 = RefineBlock([2 * self.ngf], 2 * self.ngf, act=act, start=True)
        self.refine2 = RefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, act=act)
        self.refine3 = RefineBlock([2 * self.ngf, 2 * self.ngf], self.ngf, act=act)
        self.refine4 = RefineBlock([self.ngf, self.ngf], self.ngf, act=act, end=True)

        self.control_net = MLP(input_dim, output_dim, layers)

    def _compute_cond_module(self, module, x):
        for m in module:
            x = m(x)
        return x

    def forward(self, x, u, y):
        if not self.logit_transform and not self.rescaled:
            h = 2 * x - 1.
        else:
            h = x

        output = self.begin_conv(h)

        layer1 = self._compute_cond_module(self.res1, output)
        layer2 = self._compute_cond_module(self.res2, layer1)
        layer3 = self._compute_cond_module(self.res3, layer2)
        layer4 = self._compute_cond_module(self.res4, layer3)

        ref1 = self.refine1([layer4], layer4.shape[2:])
        ref2 = self.refine2([layer3, ref1], layer3.shape[2:])
        ref3 = self.refine3([layer2, ref2], layer2.shape[2:])
        output = self.refine4([layer1, ref3], layer1.shape[2:])

        output = self.normalizer(output)
        output = self.act(output)
        output = self.end_conv(output)

        output_u = self.control_net(u)

        used_sigmas = self.sigmas[y].view(x.shape[0], *([1] * len(x.shape[1:])))
        used_sigmas_u = self.sigmas[y].view(u.shape[0], *([1] * len(u.shape[1:])))

        output = output / used_sigmas
        output_u = output_u / used_sigmas_u

        return output, output_u