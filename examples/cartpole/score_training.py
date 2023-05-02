import os
from typing import Union

import hydra
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

from score_po.score_matching import (
    ScoreEstimatorXu,
    ScoreEstimatorXux,
    langevin_dynamics,
)
from score_po.nn import MLP, TrainParams, save_module, Normalizer

OmegaConf.register_new_resolver("np.pi", lambda x: np.pi * x)


def get_score_network(xu: bool = True):
    """
    Args:
      xu: If set to true, we only compute the score of (x, u); otherwise we compute the
      score of (x, u, x_next)
    """
    if xu:
        return MLP(5, 5, [128, 128, 128, 128], activation=torch.nn.ReLU())
    else:
        return MLP(
            9, 9, [128, 128, 128, 128, 64], activation=torch.nn.ELU(), layer_norm=True
        )


def draw_score_result(
    score_estimator: Union[ScoreEstimatorXu, ScoreEstimatorXux],
    device: str,
    epsilon: float,
    steps: int,
    x_lb: torch.Tensor,
    x_ub: torch.Tensor,
):
    batch_size = 100
    x0 = score_estimator.x_normalizer.denormalize(
        2 * torch.randn((batch_size, 4), device=device)
    )
    u0 = score_estimator.u_normalizer.denormalize(
        2 * torch.randn((batch_size, 1), device=device)
    )
    xnext0 = score_estimator.x_normalizer.denormalize(
        2 * torch.randn((batch_size, 4), device=device)
    )
    if isinstance(score_estimator, ScoreEstimatorXu):
        z0 = torch.cat((x0, u0), dim=1)
    elif isinstance(score_estimator, ScoreEstimatorXux):
        z0 = torch.cat((x0, u0, xnext0), dim=1)

    z_history = langevin_dynamics(z0, score_estimator, epsilon, steps, noise=False)

    zT_np = z_history[-1].cpu().detach().numpy()

    x0_np = x0.cpu().detach().numpy()
    u0_np = u0.cpu().detach().numpy()
    xnext0_np = xnext0.cpu().detach().numpy()
    xT = zT_np[:, :4]
    uT = zT_np[:, 5:6]

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    if isinstance(score_estimator, ScoreEstimatorXu):
        # Only plot x as a point.
        ax1.scatter(x0_np[:, 0], x0_np[:, 1], color="g", label="x0")
        ax1.scatter(xT[:, 0], xT[:, 1], color="b", label="xT")
        ax2.scatter(x0_np[:, 2], x0_np[:, 3], color="g", label="x0")
        ax2.scatter(xT[:, 2], xT[:, 3], color="b", label="xT")
    elif isinstance(score_estimator, ScoreEstimatorXux):
        xnextT = zT_np[:, -4:]

        # Plot (x, x_next) as a short line segment.
        lc0 = mc.LineCollection(
            [
                [(x0_np[i, 0], x0_np[i, 1]), (xnext0_np[i, 0], xnext0_np[i, 1])]
                for i in range(batch_size)
            ],
            color="g",
            label="x0",
        )
        ax1.add_collection(lc0)
        lcT = mc.LineCollection(
            [
                [(xT[i, 0], xT[i, 1]), (xnextT[i, 0], xnextT[i, 1])]
                for i in range(batch_size)
            ],
            color="b",
            label="xT",
        )
        ax1.add_collection(lcT)

        lc0 = mc.LineCollection(
            [
                [(x0_np[i, 2], x0_np[i, 3]), (xnext0_np[i, 2], xnext0_np[i, 3])]
                for i in range(batch_size)
            ],
            color="g",
            label="x0",
        )
        ax2.add_collection(lc0)
        lcT = mc.LineCollection(
            [
                [(xT[i, 2], xT[i, 3]), (xnextT[i, 2], xnextT[i, 3])]
                for i in range(batch_size)
            ],
            color="b",
            label="xT",
        )
        ax2.add_collection(lcT)

    x_lb_np = x_lb.cpu().detach().numpy()
    x_ub_np = x_ub.cpu().detach().numpy()
    ax1.plot(
        [x_lb_np[0], x_ub_np[0], x_ub_np[0], x_lb_np[0], x_lb_np[0]],
        [x_lb_np[1], x_lb_np[1], x_ub_np[1], x_ub_np[1], x_lb_np[1]],
        linestyle="--",
        linewidth=6,
        color="r",
    )
    ax2.plot(
        [x_lb_np[2], x_ub_np[2], x_ub_np[2], x_lb_np[2], x_lb_np[2]],
        [x_lb_np[3], x_lb_np[3], x_ub_np[3], x_ub_np[3], x_lb_np[3]],
        linestyle="--",
        linewidth=6,
        color="r",
    )
    ax1.legend()
    ax2.legend()
    ax1.set_xlabel("x")
    ax1.set_ylabel(r"$\theta$")
    ax2.set_xlabel(r"$\dot{x}$")
    ax2.set_ylabel(r"$\dot{\theta}$")

    fig.tight_layout()
    sigma_val = "0:.4f".format(score_estimator.sigma[0].item())
    fig.savefig(
        os.path.join(
            os.getcwd(),
            f"score_langevin_steps{steps}_eps{epsilon}_sigma{sigma_val}.png",
        ),
        format="png",
    )


@hydra.main(config_path="./config", config_name="score_training")
def main(cfg: DictConfig):
    """
    Train a score function estimator for log p(x, u)
    """
    OmegaConf.save(cfg, os.path.join(os.getcwd(), "config.yaml"))
    torch.manual_seed(cfg.seed)
    device = cfg.device
    dataset = torch.load(cfg.dataset.load_path)

    network: MLP = get_score_network(cfg.train.xu)
    network.to(device)

    params = TrainParams()
    params.load_from_config(cfg)
    params.save_best_model = os.path.join(os.getcwd(), params.save_best_model)

    x_lo = torch.tensor(cfg.plant_param.x_lo)
    x_up = torch.tensor(cfg.plant_param.x_up)

    x_normalizer = Normalizer(k=(x_up - x_lo) / 2, b=(x_up + x_lo) / 2)
    u_normalizer = Normalizer(
        k=torch.tensor([cfg.plant_param.u_max]), b=torch.tensor([0])
    )
    if cfg.train.xu:
        estimator_cls = ScoreEstimatorXu
    else:
        estimator_cls = ScoreEstimatorXux
    score_estimator = estimator_cls(
        dim_x=4,
        dim_u=1,
        network=network,
        x_normalizer=x_normalizer,
        u_normalizer=u_normalizer,
    )
    # TODO(hongkai.dai): do a sweep on sigma (try sigma = 0.1)
    sigma = torch.tensor([cfg.train.sigma], device=device)
    loss_lst = score_estimator.train_network(dataset, params, sigma, split=True)
    draw_score_result(
        score_estimator, cfg.device, epsilon=1e-2, steps=5000, x_lb=x_lo, x_ub=x_up
    )


if __name__ == "__main__":
    main()
