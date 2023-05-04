"""
Learn the score log p(x, u) or log p(x, u, x_next)

If we learn log p(x, u), then we use ScoreEstimatorXu.
If we learn log p(x, u, x_next), then we use NoiseConditionedScoreEstimatorXux. I have
tried ScoreEstimatorXux but that doesn't learn good score function, I suspect it is
because the dynamics constraint x_next = f(x, u) is not well captured unless we
schedule the noise level.
"""
import os
from typing import Union, Optional

import hydra
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

from score_po.score_matching import (
    ScoreEstimatorXu,
    ScoreEstimatorXux,
    NoiseConditionedScoreEstimatorXux,
    langevin_dynamics,
    noise_conditioned_langevin_dynamics,
)
from score_po.nn import (
    MLP,
    MLPwEmbedding,
    TrainParams,
    save_module,
    Normalizer,
    generate_cosine_schedule,
)
from examples.cartpole.cartpole_plant import CartpolePlant, CartpoleNNDynamicalSystem

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
        return MLPwEmbedding(
            9, 9, [4096] * 4, embedding_size=10, activation=torch.nn.ELU()
        )


def draw_score_result(
    score_estimator: Union[ScoreEstimatorXu, NoiseConditionedScoreEstimatorXux],
    plant: CartpolePlant,
    nn_plant: Optional[CartpoleNNDynamicalSystem],
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
    with torch.no_grad():
        if isinstance(score_estimator, ScoreEstimatorXu):
            z0 = torch.cat((x0, u0), dim=1)
            z_history = langevin_dynamics(
                z0, score_estimator, epsilon, steps, noise=False
            )
        elif isinstance(score_estimator, NoiseConditionedScoreEstimatorXux):
            z0 = torch.cat((x0, u0, xnext0), dim=1)
            z_history = noise_conditioned_langevin_dynamics(
                z0, score_estimator, epsilon, steps, noise=False
            )

    zT = z_history[-1]
    zT_np = z_history[-1].cpu().detach().numpy()

    x0_np = x0.cpu().detach().numpy()
    u0_np = u0.cpu().detach().numpy()
    xnext0_np = xnext0.cpu().detach().numpy()
    xT = zT_np[:, :4]
    uT = zT_np[:, 4:5]
    xnext_gt = plant.dynamics_batch(zT_np[:, :4], zT_np[:, 4:5])
    if nn_plant is not None:
        xnext_nn = nn_plant.dynamics_batch(zT[:, :4], zT[:, 4:5]).cpu().detach().numpy()

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    if isinstance(score_estimator, ScoreEstimatorXu):
        # Only plot x as a point.
        ax1.scatter(x0_np[:, 0], x0_np[:, 1], color="g", label="x0")
        ax1.scatter(xT[:, 0], xT[:, 1], color="b", label="xT")
        ax2.scatter(x0_np[:, 2], x0_np[:, 3], color="g", label="x0")
        ax2.scatter(xT[:, 2], xT[:, 3], color="b", label="xT")
    elif isinstance(score_estimator, NoiseConditionedScoreEstimatorXux):
        xnextT = zT_np[:, -4:]

        # Plot (x, x_next) as a short line segment.
        lc_gt = mc.LineCollection(
            [
                [(xT[i, 0], xT[i, 1]), (xnext_gt[i, 0], xnext_gt[i, 1])]
                for i in range(batch_size)
            ],
            color="g",
            label="x_gt",
        )
        ax1.add_collection(lc_gt)
        lc_score = mc.LineCollection(
            [
                [(xT[i, 0], xT[i, 1]), (xnextT[i, 0], xnextT[i, 1])]
                for i in range(batch_size)
            ],
            color="k",
            label="x_score",
        )
        ax1.add_collection(lc_score)
        if nn_plant is not None:
            lc_nn = mc.LineCollection(
                [
                    [(xT[i, 0], xT[i, 1]), (xnext_nn[i, 0], xnext_nn[i, 1])]
                    for i in range(batch_size)
                ],
                color="r",
                label="x_nn",
            )
            ax1.add_collection(lc_nn)

        lc_gt = mc.LineCollection(
            [
                [(xT[i, 2], xT[i, 3]), (xnext_gt[i, 2], xnext_gt[i, 3])]
                for i in range(batch_size)
            ],
            color="g",
            label="x_gt",
        )
        ax2.add_collection(lc_gt)
        lc_score = mc.LineCollection(
            [
                [(xT[i, 2], xT[i, 3]), (xnextT[i, 2], xnextT[i, 3])]
                for i in range(batch_size)
            ],
            color="k",
            label="x_score",
        )
        ax2.add_collection(lc_score)
        if nn_plant is not None:
            lc_nn = mc.LineCollection(
                [
                    [(xT[i, 2], xT[i, 3]), (xnext_nn[i, 2], xnext_nn[i, 3])]
                    for i in range(batch_size)
                ],
                color="r",
                label="x_nn",
            )
            ax2.add_collection(lc_nn)

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
    if isinstance(score_estimator, ScoreEstimatorXu):
        sigma_val = "0:.4f".format(score_estimator.sigma[0].item())
    elif isinstance(score_estimator, NoiseConditionedScoreEstimatorXux):
        sigma_val = "0:.4f".format(score_estimator.sigma_lst[0])
    fig.savefig(
        os.path.join(
            os.getcwd(),
            f"score_langevin_steps{steps}_eps{epsilon}_sigma{sigma_val}.png",
        ),
        format="png",
    )

    fig_error, ax_error = plot_xux_dynamics_error_histogram(plant, nn_plant, z_history[-1])
    fig_error.savefig(
        os.path.join(
            os.getcwd(),
            f"dynamics_error_steps{steps}_eps{epsilon}_sigma{sigma_val}.png",
        ),
        format="png",
    )

    # Draw histogram of u
    fig_u = plt.figure()
    ax_u = fig_u.add_subplot()
    ax_u.hist(uT)
    ax_u.set_title("Histogram of u")
    fig_u.savefig(
        os.path.join(os.getcwd(), f"u_hist_steps{steps}_eps{epsilon}.png"), format="png"
    )


def plot_xux_dynamics_error_histogram(
    plant: CartpolePlant,
    nn_plant: Optional[CartpoleNNDynamicalSystem],
    xux_score: torch.Tensor,
):
    """
    Given a batch of (x, u, x_next), plot the histogram of the error x_next - f(x, u)
    """
    x = xux_score[:, :4]
    u = xux_score[:, 4:5]
    x_next_score = xux_score[:, -4:]
    x_next_gt = plant.dynamics_batch(x, u)
    error_score = x_next_gt - x_next_score

    if nn_plant is not None:
        x_next_nn = nn_plant.dynamics_batch(x, u)
        error_nn = x_next_gt - x_next_nn

    fig_hist = plt.figure()
    if nn_plant is None:
        ax_score = fig_hist.add_subplot()
    else:
        ax_score = fig_hist.add_subplot(211)
        ax_nn = fig_hist.add_subplot(212)
    ax_score.hist(
        torch.sqrt(torch.einsum("bi,bi->b", error_score, error_score))
        .cpu()
        .detach()
        .numpy()
    )
    ax_score.set_title("Dynamics error for score-based model")
    if nn_plant is not None:
        ax_nn.hist(
            torch.sqrt(torch.einsum("bi,bi->b", error_nn, error_nn))
            .cpu()
            .detach()
            .numpy()
        )
        ax_nn.set_title("Dynamics error for regression model")

    return fig_hist, ax_score


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
        estimator_cls = NoiseConditionedScoreEstimatorXux
    score_estimator = estimator_cls(
        dim_x=4,
        dim_u=1,
        network=network,
        x_normalizer=x_normalizer,
        u_normalizer=u_normalizer,
    )
    plant = CartpolePlant(dt=cfg.plant_param.dt)
    if cfg.train.xu:
        sigma = torch.tensor([cfg.train.sigma], device=device)
    else:
        sigma = generate_cosine_schedule(
            cfg.train.sigma_max, cfg.train.sigma_min, cfg.train.sigma_steps
        )
    loss_lst = score_estimator.train_network(dataset, params, sigma, split=True)
    draw_score_result(
        score_estimator,
        plant,
        None,
        cfg.device,
        epsilon=1e-2,
        steps=5000,
        x_lb=x_lo,
        x_ub=x_up,
    )


if __name__ == "__main__":
    main()
