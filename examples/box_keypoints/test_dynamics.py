import torch
import numpy as np
import pickle
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig

from score_po.dynamical_system import NNDynamicalSystem
from score_po.nn import MLP, train_network, TrainParams, Normalizer


def visualize_demonstration(x_now, u_now, x_next, plot_name):
    plt.figure()

    circle = plt.Circle(x_now[10:12], radius=0.035, edgecolor="k", fill=False)
    plt.gca().add_patch(circle)

    circle = plt.Circle(x_next[10:12], radius=0.035, edgecolor="r", fill=False)
    plt.gca().add_patch(circle)

    plt.arrow(x_now[10], x_now[11], u_now[0], u_now[1])

    keypts_now = x_now[:10].reshape(5, 2)
    plt.plot(keypts_now[:, 0], keypts_now[:, 1], "ko")

    keypts_next = x_next[:10].reshape(5, 2)
    plt.plot(keypts_next[:, 0], keypts_next[:, 1], "ro")

    plt.xlim([0.4, 0.9])
    plt.ylim([-0.5, 0.5])
    plt.axis("equal")
    plt.savefig(plot_name)
    plt.close()


@hydra.main(config_path="./config", config_name="dynamics_training")
def main(cfg: DictConfig):
    with open(cfg.dataset_dir, "rb") as f:
        dataset = pickle.load(f)

    dim_x = 12
    dim_u = 2
    x = dataset.tensors[0]
    u = dataset.tensors[1]
    xnext = dataset.tensors[2]
    network = MLP(dim_x + dim_u, dim_x, 4 * [1024], layer_norm=True)

    nn_dynamics = NNDynamicalSystem(
        dim_x=dim_x,
        dim_u=dim_u,
        network=network,
    )
    nn_dynamics.load_state_dict(torch.load(cfg.dynamic_weights))

    test_ind = 5

    x_now = x[test_ind]
    u_now = u[test_ind]
    x_next = xnext[test_ind]
    x_pred = nn_dynamics.dynamics(x_now, u_now)

    visualize_demonstration(x_now.numpy(), u_now.numpy(), x_next.numpy(), "real.png")
    visualize_demonstration(
        x_now.numpy(), u_now.numpy(), x_pred.detach().numpy(), "pred.png"
    )


if __name__ == "__main__":
    main()
