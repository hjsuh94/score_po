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
from score_po.nn import MLP, TrainParams
from score_po.trajectory_optimizer import (
    TrajectoryOptimizer,
    TrajectoryOptimizerSS, TrajectoryOptimizerSSParams)
from score_po.trajectory import SSTrajectory
from score_po.costs import QuadraticCost
from score_po.dynamical_system import NNDynamicalSystem

from examples.light_dark.dynamics import SingleIntegrator
from examples.light_dark.environment import Environment

@hydra.main(config_path="../config", config_name="trajopt_ss")
def main(cfg: DictConfig):
    # 1. Set up parameters.
    params = TrajectoryOptimizerSSParams()
    
    # 2. Load score function.
    network = MLP(4, 4, cfg.nn_layers)
    sf = ScoreEstimatorXu(2, 2, network)
    sf.load_state_dict(torch.load(os.path.join(
        get_original_cwd(), cfg.sf_path
    )))
    params.sf = sf
    
    # 3. Load dynamics.
    network = MLP(4, 2, cfg.nn_layers)
    ds = NNDynamicalSystem(2, 2, network)
    ds.load_state_dict(torch.load(os.path.join(
        get_original_cwd(), "examples/light_dark/weights/dynamics.pth"
    )))
    params.ds = ds
    
    # 3. Load costs.
    cost = QuadraticCost()
    cost.load_from_config(cfg)
    params.cost = cost
    
    # 4. Set up trajectory.
    trj = SSTrajectory(2, 2, cfg.trj.T,
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
        for t in range(cfg.trj.T):
            plt.arrow(x_trj[t,0], x_trj[t,1], u_trj[t,0], u_trj[t, 1], color=colormap(
            t / cfg.trj.T))
        circle = Circle([0, 0], 0.4, fill=True, color='k')
        plt.gca().add_patch(circle)
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
                     filename="{:04d}.png".format(iter))
            
    # 5. Run 
    optimizer = TrajectoryOptimizerSS(params)
    optimizer.iterate(callback)

    true_dyn = SingleIntegrator()
    params.ds = true_dyn
    params.trj = optimizer.params.trj
    optimizer = TrajectoryOptimizerSS(params)
    x_trj, u_trj = optimizer.rollout_trajectory()
    x_trj = x_trj.detach().cpu().numpy()
    u_trj = u_trj.detach().cpu().numpy()
    plot_trj(x_trj, u_trj, filename="true_trj.png", colormap="autumn")
    


if __name__ == "__main__":
    main()