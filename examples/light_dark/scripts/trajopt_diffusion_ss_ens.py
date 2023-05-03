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
    TrajectoryOptimizerSSEnsemble,
    TrajectoryOptimizerSSEnsembleParams)
from score_po.trajectory import SSTrajectory
from score_po.costs import QuadraticCost

from examples.light_dark.dynamics import SingleIntegrator
from examples.light_dark.environment import Environment

@hydra.main(config_path="../config", config_name="trajopt_ss")
def main(cfg: DictConfig):
    # 1. Set up parameters.
    params = TrajectoryOptimizerSSEnsembleParams()
    
    # 2. Load score function.
    u_normalizer = Normalizer(k =0.1 * torch.ones(2), b= torch.zeros(2))
    ds_lst = []
    for i in range(5):
        network = MLP(4, 2, [128, 128, 128])
        dynamics = NNDynamicalSystem(2, 2, network=network, u_normalizer=u_normalizer)
        ds_lst.append(dynamics)
    dynamics = NNEnsembleDynamicalSystem(dim_x=2, dim_u=2, ds_lst=ds_lst,
                                         u_normalizer=u_normalizer)
    dynamics.load_ensemble("dynamics_ensemble.pth")
    params.ds = dynamics
    
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
    
    def callback(self, loss: float, iter: int):
        if iter % cfg.plot_period == 0:
            x_trj, u_trj = self.rollout_trajectory()
            x_trj = x_trj.detach().cpu().numpy()
            u_trj = u_trj.detach().cpu().numpy()
            
            colormap = get_cmap('winter')
            
            plt.figure(figsize=(8,8))
            for k in range(self.K):
                for t in range(cfg.trj.T + 1):
                    plt.plot(x_trj[k,t,0], x_trj[k,t,1], marker='o', color=colormap(
                        t / (cfg.trj.T + 1)
                    ))
                for t in range(cfg.trj.T):
                    plt.arrow(x_trj[k,t,0], x_trj[k,t,1], u_trj[k,t,0], 
                              u_trj[k,t,1], color=colormap(
                    t / cfg.trj.T))
            circle = Circle([0, 0], 0.4, fill=True, color='k')
            plt.gca().add_patch(circle)
            plt.axis('equal')
            plt.xlim([-1, 1])
            plt.ylim([-1, 1])            
            plt.savefig("{:04d}.png".format(iter))
            plt.close()
        
    # 5. Run 
    optimizer = TrajectoryOptimizerSSEnsemble(params)
    optimizer.iterate(callback)


if __name__ == "__main__":
    main()