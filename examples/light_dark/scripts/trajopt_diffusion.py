import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.pyplot import get_cmap
import os

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from score_po.score_matching import ScoreEstimatorXux
from score_po.nn import MLP, TrainParams
from score_po.trajectory_optimizer import (
    TrajectoryOptimizer,
    TrajectoryOptimizerSF, TrajectoryOptimizerSFParams)
from score_po.trajectory import BVPTrajectory
from score_po.costs import QuadraticCost

from examples.light_dark.dynamics import SingleIntegrator
from examples.light_dark.environment import Environment

@hydra.main(config_path="../config", config_name="trajopt_diffusion")
def main(cfg: DictConfig):
    # 1. Set up parameters.
    params = TrajectoryOptimizerSFParams()
    
    # 2. Load score function.
    network = MLP(6, 6, cfg.nn_layers)
    sf = ScoreEstimatorXux(2, 2, network)
    sf.load_state_dict(torch.load(os.path.join(
        get_original_cwd(), cfg.sf_path
    )))
    params.sf = sf
    
    # 3. Load costs.
    cost = QuadraticCost()
    cost.load_from_config(cfg)
    params.cost = cost
    
    # 4. Set up trajectory.
    trj = BVPTrajectory(2, 2, cfg.trj.T,
                        torch.Tensor(cfg.trj.x0), torch.Tensor(cfg.trj.xT))
    params.trj = trj
    
    # 4. Set up optimizer
    params.load_from_config(cfg)
    params.to_device(cfg.trj.device)
    
    # 5. Define callback function
    
    def callback(params: TrajectoryOptimizerSFParams, loss: float, iter: int):
        if iter % cfg.plot_period == 0:
            x_trj, u_trj = params.trj.get_full_trajectory()
            x_trj = x_trj.detach().cpu().numpy()
            u_trj = u_trj.detach().cpu().numpy()
            
            colormap = get_cmap('winter')
            
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
            plt.savefig("{:04d}.png".format(iter))
            plt.close()
        
    # 5. Run 
    optimizer = TrajectoryOptimizerSF(params)
    optimizer.iterate(callback)


if __name__ == "__main__":
    main()