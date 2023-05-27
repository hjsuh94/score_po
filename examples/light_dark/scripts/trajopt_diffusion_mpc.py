import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.pyplot import get_cmap
import os
from tqdm import tqdm

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from score_po.score_matching import NoiseConditionedScoreEstimatorXux
from score_po.nn import MLPwEmbedding, TrainParams
from score_po.trajectory_optimizer import (
    TrajectoryOptimizer,
    TrajectoryOptimizerNCSF,
    TrajectoryOptimizerSFParams,
)
from score_po.trajectory import IVPTrajectory
from score_po.costs import QuadraticCost
from score_po.mpc import MPC

from examples.light_dark.dynamics import SingleIntegrator
from examples.light_dark.environment import Environment


@hydra.main(config_path="../config", config_name="trajopt_diffusion")
def main(cfg: DictConfig):
    # 1. Set up parameters.
    params = TrajectoryOptimizerSFParams()

    # 2. Load score function.
    network = MLPwEmbedding(6, 6, 4 * [1024], 10)
    sf = NoiseConditionedScoreEstimatorXux(2, 2, network)
    sf.load_state_dict(
        torch.load(
            os.path.join(get_original_cwd(), "examples/light_dark/weights/ncsf_xux.pth")
        )
    )
    params.sf = sf

    # 3. Load costs.
    cost = QuadraticCost()
    cost.load_from_config(cfg)
    params.cost = cost

    # 4. Set up trajectory.
    trj = IVPTrajectory(2, 2, cfg.trj.T, torch.Tensor(cfg.trj.x0))
    params.trj = trj

    # 4. Set up optimizer
    params.load_from_config(cfg)
    params.to_device(cfg.trj.device)

    # 5. Define callback function

    # 5. Run
    optimizer = TrajectoryOptimizerNCSF(params)

    # 6. Run MPC
    true_dynamics = SingleIntegrator()
    mpc = MPC(optimizer)
    T = 100
    x_history = torch.zeros(T + 1, 2)
    u_history = torch.zeros(T, 2)
    x_history[0] = torch.Tensor([0.8, 0.2])
    for t in tqdm(range(T)):
        u_history[t] = mpc.get_action(x_history[t])
        x_history[t + 1] = true_dynamics.dynamics(x_history[t], u_history[t])
        x_trj, u_trj = mpc.opt.trj.get_full_trajectory()
        x_trj = x_trj.cpu().detach().numpy()
        u_trj = u_trj.cpu().detach().numpy()
        colormap = get_cmap("winter")

        plt.figure(figsize=(8, 8))
        opt_T = mpc.opt.trj.T
        for tp in range(opt_T + 1):
            plt.plot(
                x_trj[tp, 0], x_trj[tp, 1], marker="o", color=colormap(tp / (opt_T + 1))
            )
        for tp in range(opt_T):
            plt.arrow(
                x_trj[tp, 0],
                x_trj[tp, 1],
                u_trj[tp, 0],
                u_trj[tp, 1],
                color=colormap(tp / (opt_T + 1)),
            )

        plt.plot(
            x_history[: t + 1, 0].detach().numpy(),
            x_history[: t + 1, 1].detach().numpy(),
            "ro-",
        )
        circle = Circle([0, 0], 0.4, fill=False, color="k")
        plt.gca().add_patch(circle)
        plt.axis("equal")
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.savefig("{:04d}.png".format(t))
        plt.close()


if __name__ == "__main__":
    main()
