import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.pyplot import get_cmap
import os

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from score_po.score_matching import ScoreEstimatorXu
from score_po.nn import MLP, Normalizer
from score_po.trajectory_optimizer import (
    TrajectoryOptimizer,
    TrajectoryOptimizerSS, TrajectoryOptimizerSSParams)
from score_po.trajectory import SSTrajectory
from score_po.costs import QuadraticCost
from score_po.dynamical_system import NNDynamicalSystem

from examples.pendulum.pendulum_keypoint_plant import PendulumPlant

@hydra.main(config_path="./config", config_name="trajopt_diffusion")
def main(cfg: DictConfig):
    # 1. Set up parameters.
    params = TrajectoryOptimizerSSParams()
    plant = PendulumPlant(dt=cfg.plant_param.dt)
    x_lo = torch.tensor( cfg.plant_param.x_lo)
    x_up = torch.tensor(cfg.plant_param.x_up)
    x_kp_lo = torch.tensor([-plant.l, -plant.l, x_lo[1]])
    x_kp_up = torch.tensor([plant.l, plant.l, x_up[1]])
    u_lo = torch.tensor([-cfg.plant_param.u_max])
    u_up = torch.tensor([cfg.plant_param.u_max])
    x_normalizer = Normalizer(k=(x_kp_up - x_kp_lo) / 2, b=(x_kp_up + x_kp_lo) / 2)
    u_normalizer = Normalizer(k=(u_up - u_lo) / 2, b=(u_up + u_lo) / 2)
    
    # 2. Load score function.
    network = MLP(4, 4, cfg.nn_layers)
    sf = ScoreEstimatorXu(3, 1, network)
    sf.load_state_dict(torch.load(os.path.join(
        get_original_cwd(), "examples/pendulum/weights/sf_xu.pth"
    )))
    params.sf = sf
    
    # 3. Load dynamics.
    network = MLP(4, 3, cfg.nn_layers, layer_norm=True)
    ds = NNDynamicalSystem(3, 1, network, x_normalizer=x_normalizer, u_normalizer=u_normalizer)
    ds.load_state_dict(torch.load(os.path.join(
        get_original_cwd(), "examples/pendulum/weights/dynamics.pth"
    )))
    params.ds = ds
    
    # 3. Load costs.
    cost = QuadraticCost()
    cost.load_from_config(cfg)
    params.cost = cost
    
    # 4. Set up trajectory.
    trj = SSTrajectory(3, 1, cfg.trj.T,
                        torch.Tensor(cfg.trj.x0))
    params.trj = trj
    
    # 4. Set up optimizer
    params.load_from_config(cfg)
    params.to_device(cfg.trj.device)
    
    # 5. Define callback function
    
    def plot_trj(x_trj, u_trj, filename="trj.png", colormap='winter'):
        colormap = get_cmap(colormap)
        plt.figure(figsize=(8,8))
        for t in range(cfg.trj.T + 1):
            plt.plot(x_trj[t,0], x_trj[t,1], marker='o', color=colormap(
                t / (cfg.trj.T + 1)
            ))

        theta = np.linspace(0, 2 * np.pi, 100)
        plt.plot(np.cos(theta) * cfg.pendulum_length, np.sin(theta) * cfg.pendulum_length, linestyle="--")

        plt.axis('equal')
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])            
        plt.savefig(filename)
        plt.close()
    
    def callback(self, loss: float, iter: int):
        if iter % cfg.plot_period == 0:
            x_trj, u_trj = self.rollout_trajectory()
            x_trj = x_trj.detach().cpu().numpy()
            u_trj = u_trj.detach().cpu().numpy()
            plot_trj(x_trj, u_trj,
                     filename="{:03d}.png".format(iter//cfg.plot_period))
            
    # 5. Run 
    optimizer = TrajectoryOptimizerSS(params)
    optimizer.iterate(callback)

    true_dyn = PendulumPlant(dt=cfg.plant_param.dt)
    params.ds = true_dyn
    params.trj = optimizer.params.trj
    optimizer = TrajectoryOptimizerSS(params)
    x_trj, u_trj = optimizer.rollout_trajectory()
    x_trj = x_trj.detach().cpu().numpy()
    u_trj = u_trj.detach().cpu().numpy()
    plot_trj(x_trj, u_trj, filename="true_trj.png", colormap="autumn")
    


if __name__ == "__main__":
    main()