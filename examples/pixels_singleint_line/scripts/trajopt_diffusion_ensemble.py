import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.pyplot import get_cmap
import os

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from score_po.dynamical_system import (
    NNDynamicalSystem,
    NNEnsembleDynamicalSystem
)
from score_po.nn import MLP, TrainParams, Normalizer
from score_po.trajectory_optimizer import (
    TrajectoryOptimizerParams,
    TrajectoryOptimizerSSEnsembleImage,
    TrajectoryOptimizerSSEnsembleParams)
from score_po.trajectory import SSTrajectory
from score_po.costs import QuadraticCost

from examples.pixels_singleint_line.dynamics import SingleIntegratorPixels
from examples.pixels_singleint_line.environment import Environment
from examples.pixels_singleint_line.dynamics import SingleIntegratorPixelsAverage

from score_po.image_nn import NN_ImageU_Dynamics
from score_po.dynamical_system import NNDynamicalSystem_Image
from score_po.costs import QuadraticCostImage
import scipy.io as sio

from score_po.trajectory_optimizer import (
    TrajectoryOptimizer, TrajectoryOptimizerNCSS,
    TrajectoryOptimizerNCSS_Image, TrajectoryOptimizerSSParams)
from score_po.trajectory import SSTrajectory
from score_po.costs import QuadraticCost
from score_po.dynamical_system import NNDynamicalSystem

from examples.pixels_singleint_line.dynamics import SingleIntegratorPixelsAverage
from examples.pixels_singleint_line.environment import Environment
from score_po.nn import NCSN_ImageU, Normalizer, get_sigmas
from score_po.score_matching import NoiseConditionedScoreEstimatorXuImageU
from score_po.image_nn import NN_ImageU_Dynamics
from score_po.dynamical_system import NNDynamicalSystem_Image
from score_po.costs import QuadraticCostImage

@hydra.main(config_path="../config", config_name="trajopt")
def main(cfg: DictConfig):
    # 1. Set up parameters.
    params = TrajectoryOptimizerSSEnsembleParams()
    nx, nu = 32, 32

    # 2. Load score function.
    dynamics_lst = []
    for i in range(10):
        network = NN_ImageU_Dynamics(cfg).to(cfg.train.device)
        dynamics = NNDynamicalSystem_Image(int(nx ** 2), int(nu ** 2), network, x_normalizer=None, u_normalizer=None)
        dynamics_lst.append(dynamics)
    dynamics = NNEnsembleDynamicalSystem(dim_x=int(nx ** 2), dim_u=int(nu ** 2), ds_lst=dynamics_lst)

    dynamics.load_ensemble(os.path.join(
        get_original_cwd(), "../weights/dynamics.pth"
    ))
    params.ds = dynamics

    # 3. Load costs.
    env = Environment()
    cost = QuadraticCostImage()
    yd, _ = env.sample_image(1, nx, samples=torch.tensor([cfg.cost.xd]).double())
    cost.load_from_config(cfg)
    # cost.load_from_config(cfg, xd=yd[0])
    params.cost = cost
    y_mesh, x_mesh = torch.meshgrid(torch.linspace(-1.2, 1.2, 32), torch.linspace(-1.2, 1.2, 32))
    y_mesh_u, x_mesh_u = torch.meshgrid(torch.linspace(-0.3, 0.3, 32), torch.linspace(-0.3, 0.3, 32))

    # 4. Set up trajectory.
    y0, _ = env.sample_image(1, nx, samples=torch.tensor([cfg.trj.x0]))
    yT, _ = env.sample_image(1, nx, samples=torch.tensor([cfg.trj.xT]))
    trj = SSTrajectory(int(nx ** 2), int(nu ** 2), cfg.trj.T,
                       torch.Tensor(y0[0]).float())
    params.trj = trj
    params.max_iters_trunc = cfg.trj.max_iters_trunc

    # 4. Set up optimizer
    params.load_from_config(cfg)
    params.to_device(cfg.trj.device)

    # 5. Define callback function

    def plot_trj(x_trj, u_trj, iter, file_name='', colormap_name='winter'):
        if not isinstance(x_trj, np.ndarray):
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

        colormap = get_cmap(colormap_name)

        plt.figure(figsize=(8, 8))
        for t in range(cfg.trj.T + 1):
            plt.plot(x_trj[t, 0], x_trj[t, 1], marker='o', color=colormap(
                t / (cfg.trj.T + 1)
            ))
        for t in range(cfg.trj.T):
            plt.arrow(x_trj[t, 0], x_trj[t, 1], u_trj[t, 0], u_trj[t, 1], color=colormap(
                t / cfg.trj.T))
        # circle = Circle([0, 0], 0.4, fill=True, color='k')
        # plt.gca().add_patch(circle)
        plt.axis('equal')
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.savefig(file_name + "{:04d}.png".format(iter))
        plt.close()

        fig, axs = plt.subplots(4, 4)
        fig.set_figheight(8)
        fig.set_figwidth(14)
        idx = 0
        for i in range(4):
            for j in range(4):
                if idx >= x_trj_orig.shape[0]:
                    break
                axs[i, j].imshow(x_trj_orig[idx])
                axs[i, j].set_title('T = ' + str(idx))
                axs[i, j].axis('off')
                idx += 1
        plt.savefig(file_name + "x_images_{:04d}.png".format(iter))
        plt.close()

        fig, axs = plt.subplots(4, 4)
        fig.set_figheight(8)
        fig.set_figwidth(14)
        idx = 0
        for i in range(4):
            for j in range(4):
                if idx >= u_trj_orig.shape[0]:
                    break
                axs[i, j].imshow(u_trj_orig[idx].reshape(32, 32))
                axs[i, j].set_title('T = ' + str(idx))
                axs[i, j].axis('off')
                idx += 1
        plt.savefig(file_name + "u_images_{:04d}.png".format(iter))
        plt.close()
    def callback(self, loss, iter: int, x_trj, u_trj):
        if iter % cfg.plot_period == 0:
            # print('blah')
            # x_trj, u_trj = self.rollout_trajectory()
            plot_trj(x_trj[0], u_trj[0], iter)

    def plot_trj_small(x_trj, u_trj, iter, indices, file_name='', colormap_name='winter'):
        if not isinstance(x_trj, np.ndarray):
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

        fig, axs = plt.subplots(1, x_trj.shape[0], squeeze=False)
        fig.set_figheight(8)
        fig.set_figwidth(14)
        idx = 0
        for j in range(x_trj.shape[0]):
            axs[0, j].imshow(x_trj_orig[idx].reshape(32, 32))
            axs[0, j].set_title('T = ' + str(indices[j]), fontsize=25)
            axs[0, j].axis('off')
            idx += 1
        plt.savefig(file_name + "x_images_{:04d}.pdf".format(iter), format='pdf')
        plt.close()

        fig, axs = plt.subplots(1, x_trj.shape[0], squeeze=False)
        fig.set_figheight(8)
        fig.set_figwidth(14)
        idx = 0
        for j in range(x_trj.shape[0]):
            axs[0, j].imshow(u_trj_orig[idx].reshape(32, 32))
            axs[0, j].set_title('T = ' + str(indices[j]), fontsize=25)
            axs[0, j].axis('off')
            idx += 1
        plt.savefig(file_name + "u_images_{:04d}.pdf".format(iter), format='pdf')
        plt.close()

    # 5. Run
    optimizer = TrajectoryOptimizerSSEnsembleImage(params)
    optimizer.iterate(callback)
    x_trj_hal, u_trj_hal = optimizer.rollout_trajectory()
    x_norm = x_trj_hal[0].clamp(min=0.0, max=1.) / (x_trj_hal[0].clamp(min=0.0, max=1.) + 1e-6).sum(dim=(1, 2))[:, None,
                                                   None].repeat(1, x_trj_hal[0].shape[-2], x_trj_hal[0].shape[-1])
    pos_x = (x_norm * x_mesh.to(x_norm.device)).sum(dim=(1, 2))
    pos_y = (x_norm * y_mesh.to(x_norm.device)).sum(dim=(1, 2))
    x_batch = torch.hstack([pos_x[:, None], pos_y[:, None]])

    # True trajectory
    # Average out the trajectory over the ensemble
    # Rollout the trajectory on the true dynamics
    true_dyn = SingleIntegratorPixelsAverage()
    params = TrajectoryOptimizerSSParams()
    nx, nu = 32, 32

    # 2. Load score function.
    network = NCSN_ImageU(cfg)
    u_normalizer = Normalizer(k=0.2 * torch.ones(2), b=torch.zeros(2))
    sf = NoiseConditionedScoreEstimatorXuImageU(int(nx ** 2), nu, get_sigmas(cfg), network,
                                                 x_normalizer=None, u_normalizer=u_normalizer)
    sf.load_state_dict(torch.load(os.path.join(
        get_original_cwd(), "../weights/ncsn_xu.pth"
    )))
    params.sf = sf

    # 3. Load dynamics.
    network = NN_ImageU_Dynamics(cfg).to(cfg.train.device)
    ds = NNDynamicalSystem_Image(int(nx ** 2), int(nu ** 2), network, x_normalizer=None, u_normalizer=None)
    ds.load_state_dict(torch.load(os.path.join(
        get_original_cwd(), "../weights/dynamics_00.pth"
    )))
    params.ds = ds

    # 3. Load costs.
    env = Environment()
    cost = QuadraticCostImage()
    yd, _ = env.sample_image(1, nx, samples=torch.tensor([cfg.cost.xd]).double())
    cost.load_from_config(cfg)
    # cost.load_from_config(cfg, xd=yd[0])
    params.cost = cost
    y_mesh, x_mesh = torch.meshgrid(torch.linspace(-1.2, 1.2, 32), torch.linspace(-1.2, 1.2, 32))
    y_mesh_u, x_mesh_u = torch.meshgrid(torch.linspace(-0.3, 0.3, 32), torch.linspace(-0.3, 0.3, 32))

    # 4. Set up trajectory.
    y0, _ = env.sample_image(1, nx, samples=torch.tensor([cfg.trj.x0]))
    yT, _ = env.sample_image(1, nx, samples=torch.tensor([cfg.trj.xT]))
    trj = SSTrajectory(int(nx**2), int(nu**2), cfg.trj.T,
                       torch.Tensor(y0[0]).float())
    params.trj = trj
    params.max_iters_trunc = cfg.trj.max_iters_trunc

    # 4. Set up optimizer
    params.load_from_config(cfg)
    params.to_device(cfg.trj.device)


    params.ds = true_dyn
    params.trj = optimizer.params.trj
    params.trj.x0 = torch.tensor(cfg.trj.x0).to(cfg.trj.device)
    params.trj.u_trj = torch.nn.Parameter(u_trj_hal.mean(dim=0))
    optimizer = TrajectoryOptimizerNCSS_Image(params)
    x_trj, u_trj = optimizer.rollout_trajectory_2d()
    y_rollout, _ = env.sample_image(x_trj.shape[0], nx, samples=x_trj)
    plot_trj(y_rollout, u_trj, 0, file_name="plan_rerender.png", colormap_name="autumn")

    u_batch = u_trj
    u_batch = u_batch.reshape(u_batch.shape[0], 32, 32)
    eps = 1e-6
    u_norm = u_batch.clamp(min=0.0, max=1.) / (u_batch.clamp(min=0.0, max=1.) + eps).sum(dim=(1, 2))[:, None,
                                              None].repeat(
        1, u_batch.shape[-2], u_batch.shape[-1])
    pos_x = (u_norm * x_mesh_u.to(u_norm.device)).sum(dim=(1, 2))
    pos_y = (u_norm * y_mesh_u.to(u_norm.device)).sum(dim=(1, 2))
    u_trj_2d = torch.hstack([pos_x[:, None], pos_y[:, None]])
    sio.savemat('traj_data.mat', {'x_batch': x_batch.detach().cpu().numpy(), 'x_trj': x_trj.detach().cpu().numpy(),
                                  'u_trj': u_trj.detach().cpu().numpy(), 'x_trj_hal': x_trj_hal.detach().cpu().numpy(),
                                  'u_trj_hal': u_trj_hal.detach().cpu().numpy(),
                                  'u_trj_2d': u_trj_2d.detach().cpu().numpy()})

    plot_trj_small(x_trj_hal[0][0::5], u_trj_hal[0][[0,5,10,14]], 0, file_name="hallucinated_")
    plot_trj_small(y_rollout[0::5], u_trj[[0,5,10,14]], 0, file_name="trueroll_")

    reward_actual = optimizer.cost.get_running_cost_batch(y_rollout[:-1].cuda().float(), u_trj[:]).sum() + \
                    optimizer.cost.get_terminal_cost(y_rollout[-1].cuda().float())
    reward_hallucinated = optimizer.cost.get_running_cost_batch(x_trj_hal[0][:-1].cuda().float(), u_trj_hal[:]).sum() + \
                          optimizer.cost.get_terminal_cost(x_trj_hal[0][-1].cuda().float())
    print('done')




if __name__ == "__main__":
    main()