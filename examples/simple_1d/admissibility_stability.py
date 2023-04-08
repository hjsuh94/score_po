# Is the data distance really convex?

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

import hydra
from omegaconf import DictConfig

import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from tqdm import tqdm

from score_po.nn import MLP, EnsembleNetwork, TrainParams, train_network
from score_po.data_distance import DataDistance, DataDistanceEstimator

target_fun = lambda x: x**2.0 * np.sin(x)


def get_ensembles(x, data, cfg):
    data_tensor = torch.Tensor(data).reshape(-1, 1)
    dataset = TensorDataset(data_tensor)
    label = torch.Tensor(target_fun(data)).reshape(-1, 1)
    params = TrainParams()
    params.load_from_config(cfg)
    network_lst = []
    criterion = nn.MSELoss()

    loss_fn = lambda x_batch, net: criterion(net(x_batch), label)

    for i in tqdm(range(cfg.ensemble_size)):
        net = MLP(1, 1, cfg.nn_layers)
        train_network(net, params, dataset, loss_fn, split=False)
        network_lst.append(net)

    ensemble = EnsembleNetwork(1, 1, network_lst)

    x_tensor = torch.Tensor(x).reshape(-1, 1)
    z = ensemble(x_tensor).detach().numpy()[:, 0]
    std = np.sqrt(ensemble.get_var(x_tensor).detach().numpy())
    return z, std


def get_data_energy(x, data, std):
    data_tensor = torch.Tensor(data).reshape(-1, 1)
    x_tensor = torch.Tensor(x).reshape(-1, 1)

    dataset = TensorDataset(data_tensor)
    metric = torch.ones(1) * 1 / (std**2.0)
    dst = DataDistance(dataset, metric)
    return dst.get_energy_to_data(x_tensor).cpu().numpy()


def get_gp(x, data, target_fun):
    kernel = 1.0 * RBF(length_scale=0.1, length_scale_bounds=(1e-1, 10.0))
    gp = GaussianProcessRegressor(kernel=kernel)
    gp.fit(data.reshape(-1, 1), target_fun(data).reshape(-1, 1))
    z, std = gp.predict(x.reshape(-1, 1), return_std=True)
    return z, std


@hydra.main(config_path="./config", config_name="admissibility_stability")
def main(cfg: DictConfig):

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    plt.figure(figsize=(15, 7))
    plt.tight_layout()

    # -----------------------------------------------------------------------------
    # 1. Draw data.
    # -----------------------------------------------------------------------------
    N = cfg.N
    data = 2.0 * (np.random.rand(N) - 0.5)
    plt.subplot(2, 3, 4)
    plt.title("Energy to Data")
    plt.plot(data, np.zeros(N), "ro")
    x_range = cfg.x_range
    x = np.linspace(-x_range, x_range, cfg.n_grid)

    # -----------------------------------------------------------------------------
    # 2. Fit Ensembles
    # -----------------------------------------------------------------------------

    z, std = get_ensembles(x, data, cfg)
    plt.subplot(2, 3, 3)
    plt.title("Ensembles")
    plt.plot(data, target_fun(data), "ro", label="data")
    plt.plot(x, target_fun(x), "k--", label="target")
    plt.plot(x, z, color="springgreen", label="predicted")
    scale = cfg.scale_ensemble
    plt.fill_between(
        x,
        z - scale * std,
        z + scale * std,
        alpha=0.1,
        color="black",
        label="uncertainty",
    )

    plt.legend()
    plt.ylim([cfg.ylim_bottom, cfg.ylim_top])

    plt.subplot(2, 3, 6)
    plt.title("Ensemble Variance")
    plt.plot(data, np.zeros(N), "ro")
    plt.plot(
        x,
        cfg.scale_ensemble_plot * scale * std**2.0,
        color="springgreen",
        label="Uncertainty",
    )
    plt.plot(x, (z - target_fun(x)) ** 2.0, "k--", label="squared error")
    plt.legend()

    # -----------------------------------------------------------------------------
    # 3. Evaluate data to distance
    # -----------------------------------------------------------------------------
    softmin_dist = get_data_energy(x, data, cfg.sigma)

    plt.subplot(2, 3, 1)
    plt.title("Single NN + Energy to Data")
    plt.plot(data, target_fun(data), "ro", label="data")
    plt.plot(x, target_fun(x), "k--", label="target")
    plt.plot(x, z, color="springgreen", label="predicted")

    scale = cfg.scale_softmin
    plt.fill_between(
        x,
        z - scale * np.sqrt(softmin_dist - np.min(softmin_dist)),
        z + scale * np.sqrt(softmin_dist - np.min(softmin_dist)),
        alpha=0.1,
        color="black",
        label="uncertainty",
    )

    plt.ylim([cfg.ylim_bottom, cfg.ylim_top])

    # Get distance.

    plt.subplot(2, 3, 4)
    plt.plot(x, scale * softmin_dist, color="springgreen")
    plt.plot(x, (z - target_fun(x)) ** 2.0, "k--")

    # -----------------------------------------------------------------------------
    # 4. Fit GPs.
    # -----------------------------------------------------------------------------

    # Gaussian processes training.
    z, std = get_gp(x, data, target_fun)

    plt.subplot(2, 3, 2)
    plt.title("Gaussian Processes (GP)")
    plt.plot(x, target_fun(x), "k--")
    plt.plot(data, target_fun(data), "ro")
    plt.plot(x, z, color="springgreen")
    scale = cfg.scale_gp
    plt.fill_between(
        x,
        z - scale * std,
        z + scale * std,
        alpha=0.1,
        color="black",
    )
    plt.ylim([cfg.ylim_bottom, cfg.ylim_top])

    plt.subplot(2, 3, 5)
    plt.title("GP Variance")
    plt.plot(x, cfg.scale_gp_plot * scale * std**2.0, color="springgreen")
    plt.plot(data, np.zeros(N), "ro")
    plt.plot(x, (z - target_fun(x)) ** 2.0, "k--")

    plt.savefig("test.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
