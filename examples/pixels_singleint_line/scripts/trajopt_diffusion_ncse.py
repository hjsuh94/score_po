import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.pyplot import get_cmap
import os

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from score_po.score_matching import NoiseConditionedScoreEstimatorXuxImageU
from score_po.nn import NCSN_ImageU, get_sigmas, Normalizer, tensor_linspace
from score_po.trajectory_optimizer import (
    TrajectoryOptimizer,
    TrajectoryOptimizerNCSFImageU, TrajectoryOptimizerSFParams)
from score_po.trajectory import BVPTrajectory
from score_po.costs import QuadraticCostImage

from examples.pixels_singleint_line.dynamics import SingleIntegratorPixels
from examples.pixels_singleint_line.environment import Environment


@hydra.main(config_path="../config", config_name="trajopt_diffusion")
def main(cfg: DictConfig):
    # 1. Set up parameters.
    params = TrajectoryOptimizerSFParams()
    nx = 32
    nu = 32

    # 2. Load score function.
    network = NCSN_ImageU(cfg)
    u_normalizer = Normalizer(k=0.2 * torch.ones(2), b=torch.zeros(2))
    sf = NoiseConditionedScoreEstimatorXuxImageU(int(nx**2), nu, get_sigmas(cfg), network,
                                                 x_normalizer=None, u_normalizer=u_normalizer)
    sf.load_state_dict(torch.load(os.path.join(
        get_original_cwd(), "../weights/ncsn_xux.pth"
    )))
    params.sf = sf

    # 3. Load costs.
    env = Environment()
    cost = QuadraticCostImage()
    yd, _ = env.sample_image(1, nx, samples=torch.tensor([cfg.cost.xd]).double())
    cost.load_from_config(cfg)
    #cost.load_from_config(cfg, xd=yd[0])
    params.cost = cost
    x_mesh, y_mesh = torch.meshgrid(torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 32))
    x_mesh_u, y_mesh_u = torch.meshgrid(torch.linspace(-0.2, 0.2, 32), torch.linspace(-0.2, 0.2, 32))

    # 4. Set up trajectory.
    y0, _ = env.sample_image(1, nx, samples=torch.tensor([cfg.trj.x0]))
    yT, _ = env.sample_image(1, nx, samples=torch.tensor([cfg.trj.xT]))
    trj = BVPTrajectory(int(nx**2), int(nu**2), cfg.trj.T,
                        torch.Tensor(y0[0]).float(), torch.Tensor(yT[0]).float())
    params.trj = trj

    # 4. Set up optimizer
    params.load_from_config(cfg)
    params.to_device(cfg.trj.device)

    # 5. Define callback function

    def callback(params: TrajectoryOptimizerSFParams, loss: float, iter: int):
        if iter % cfg.plot_period == 0:
            x_trj, u_trj = params.trj.get_full_trajectory()
            x_trj_orig, u_trj_orig = x_trj.detach().cpu().numpy(), u_trj.detach().cpu().numpy()

            x_norm = x_trj / x_trj.sum(dim=(1, 2))[:, None, None].repeat(1, x_trj.shape[-2], x_trj.shape[-1])
            pos_x = (x_norm * x_mesh.to(x_norm.device)).sum(dim=(1, 2))
            pos_y = (x_norm * y_mesh.to(x_norm.device)).sum(dim=(1, 2))
            x_trj = torch.hstack([pos_x[:, None], pos_y[:, None]])

            u_batch = u_trj.reshape(u_trj.shape[0], int(np.sqrt(u_trj.shape[1])), int(np.sqrt(u_trj.shape[1])))
            u_norm = u_batch / u_batch.sum(dim=(1, 2))[:, None, None].repeat(1, u_batch.shape[-2], u_batch.shape[-1])
            pos_x = (u_norm * x_mesh_u.to(u_norm.device)).sum(dim=(1, 2))
            pos_y = (u_norm * y_mesh_u.to(u_norm.device)).sum(dim=(1, 2))
            u_trj = torch.hstack([pos_x[:, None], pos_y[:, None]])

            x_trj = x_trj.detach().cpu().numpy()
            u_trj = u_trj.detach().cpu().numpy()

            colormap = get_cmap('winter')

            plt.figure(figsize=(8, 8))
            for t in range(cfg.trj.T + 1):
                plt.plot(x_trj[t, 0], x_trj[t, 1], marker='o', color=colormap(
                    t / (cfg.trj.T + 1)
                ))
            for t in range(cfg.trj.T):
                plt.arrow(x_trj[t, 0], x_trj[t, 1], u_trj[t, 0], u_trj[t, 1], color=colormap(
                    t / cfg.trj.T))
            circle = Circle([0, 0], 0.4, fill=True, color='k')
            plt.gca().add_patch(circle)
            plt.axis('equal')
            plt.xlim([-1, 1])
            plt.ylim([-1, 1])
            plt.savefig("{:04d}.png".format(iter))
            plt.close()

            fig, axs = plt.subplots(4, 7)
            fig.set_figheight(8)
            fig.set_figwidth(14)
            idx = 0
            for i in range(7):
                for j in range(4):
                    if idx >= x_trj_orig.shape[0]:
                        break
                    axs[j,i].imshow(x_trj_orig[idx])
                    axs[j,i].set_title(str(idx))
                    idx += 1
            plt.savefig("x_images_{:04d}.png".format(iter))
            plt.close()

            fig, axs = plt.subplots(4, 7)
            fig.set_figheight(8)
            fig.set_figwidth(14)
            idx = 0
            for i in range(7):
                for j in range(4):
                    if idx >= u_trj_orig.shape[0]:
                        break
                    axs[j, i].imshow(u_trj_orig[idx].reshape(32, 32))
                    axs[j, i].set_title(str(idx))
                    idx += 1
            plt.savefig("u_images_{:04d}.png".format(iter))
            plt.close()

    # 5. Run
    x_init = tensor_linspace(torch.tensor(cfg.trj.x0), torch.tensor(cfg.trj.xT), steps=cfg.trj.T-1).T
    y_init, _ = env.sample_image(cfg.trj.T-1, nx, samples=x_init)
    u_init = env.sample_control_image(nx, x_init[1:] - x_init[:-1], buf=2.)
    u_init = torch.cat([u_init, u_init[0:2]], dim=0)
    optimizer = TrajectoryOptimizerNCSFImageU(params)
    optimizer.iterate(callback, xnext_trj_init=y_init.to(cfg.trj.device).float(),
                      u_trj_init=u_init.view(u_init.shape[0], -1).to(cfg.trj.device).float())
    # optimizer.iterate(callback)


if __name__ == "__main__":
    main()
