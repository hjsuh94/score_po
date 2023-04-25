import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import os
import torch

from score_po.score_matching import ScoreEstimatorXu, langevin_dynamics
from score_po.nn import MLP, TrainParams, Normalizer
from examples.corridor.learn_model import draw_corridor


def get_score_network():
    return MLP(4, 4, [32, 32, 32, 32])


def draw_score_result(
    score_estimator: ScoreEstimatorXu,
    device: str,
    corridor_width,
    horizontal_max,
    vertical_max,
    epsilon,
    steps,
):
    # Now use Langevian dynamics to generate samples.
    x0 = torch.randn((100, 4)).to(device)
    x_history = langevin_dynamics(x0, score_estimator, epsilon, steps)

    fig = plt.figure()
    ax = fig.add_subplot()
    draw_corridor(ax, corridor_width, horizontal_max, vertical_max, color="k")
    ax.scatter(
        x0[:, 0].cpu().detach().numpy(),
        x0[:, 1].cpu().detach().numpy(),
        color="r",
        label="x0",
    )
    ax.scatter(
        x_history[-1, :, 0].cpu().detach().numpy(),
        x_history[-1, :, 1].cpu().detach().numpy(),
        color="b",
        label="xT",
    )
    ax.legend()
    fig.savefig(os.path.join(os.getcwd(), "score_langevian.png"), format="png")


@hydra.main(config_path="./config", config_name="score_training")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    device = cfg.device
    dataset = torch.load(cfg.dataset.load_path)

    network: MLP = get_score_network()
    network.to(device)

    params = TrainParams()
    params.load_from_config(cfg)

    corridor_width = cfg.corridor_width
    horizontal_max = cfg.horizontal_max
    vertical_max = cfg.vertical_max
    x_lo = torch.tensor([-corridor_width / 2, -corridor_width / 2])
    x_up = torch.tensor([horizontal_max, vertical_max])
    u_lo = torch.tensor(cfg.u_lo)
    u_up = torch.tensor(cfg.u_up)

    x_normalizer = Normalizer(k=(x_up - x_lo) / 2, b=(x_up + x_lo) / 2)
    u_normalizer = Normalizer(k=(u_up - u_lo) / 2, b=(u_up + u_lo) / 2)
    score_estimator = ScoreEstimatorXu(
        dim_x=2,
        dim_u=2,
        network=network,
        x_normalizer=x_normalizer,
        u_normalizer=u_normalizer,
    )
    loss_lst = score_estimator.train_network(dataset, params, sigma=0.01, split=True)

    # Now use Langevian dynamics to generate samples.
    draw_score_result(
        score_estimator,
        device,
        corridor_width,
        horizontal_max,
        vertical_max,
        epsilon=1e-4,
        steps=1000,
    )


if __name__ == "__main__":
    main()
