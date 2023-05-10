import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.pyplot import get_cmap
import os

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from score_po.score_matching import NoiseConditionedScoreEstimatorXux
from score_po.nn import MLPwEmbedding, TrainParams
from score_po.trajectory_optimizer import (
    TrajectoryOptimizer,
    TrajectoryOptimizerNCSF, TrajectoryOptimizerSFParams)
from score_po.trajectory import BVPTrajectory
from score_po.costs import QuadraticCost

from examples.pendulum.pendulum_keypoint_plant import PendulumPlant

@hydra.main(config_path="./config", config_name="trajopt_diffusion")
def main(cfg: DictConfig):
    # 1. Set up parameters.
    params = TrajectoryOptimizerSFParams()
    
    # 2. Load score function.
    network = MLPwEmbedding(7, 7, cfg.nn_layers, 10)
    sf = NoiseConditionedScoreEstimatorXux(3, 1, network)
    sf.load_state_dict(torch.load(os.path.join(
        get_original_cwd(), "examples/pendulum/weights/ncsn_xux.pth"
    )))
    params.sf = sf
    
    # 3. Load costs.
    cost = QuadraticCost()
    cost.load_from_config(cfg)
    params.cost = cost
    
    # 4. Set up trajectory.
    trj = BVPTrajectory(3, 1, cfg.trj.T,
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
            # for t in range(cfg.trj.T):
                # plt.arrow(x_trj[t,0], x_trj[t,1], u_trj[t,0], color=colormap(
                # t / cfg.trj.T))
            theta = np.linspace(0, 2 * np.pi, 100)
            plt.plot(np.cos(theta) * cfg.pendulum_length, np.sin(theta) * cfg.pendulum_length, linestyle="--")
            plt.axis('equal')
            plt.xlim([-1, 1])
            plt.ylim([-1, 1])            
            plt.savefig("{:03d}.png".format(iter//cfg.plot_period))
            plt.close()
        
    # 5. Run 
    if cfg.optimize_trj:
        optimizer = TrajectoryOptimizerNCSF(params)
        optimizer.iterate(callback)
    else:
        params.trj.load_state_dict(torch.load(os.path.join(
        get_original_cwd(), "examples/pendulum/weights/checkpoint.pth"
    )))


    plant = PendulumPlant(cfg.plant_param.dt)
    x_trj, u_trj = params.trj.get_full_trajectory()
    x_trj = x_trj.detach().cpu().numpy()
    x_history = torch.zeros(cfg.trj.T + 1, 2, device=cfg.trj.device)
    x_history[0] = torch.Tensor([np.pi, 0])
    x_kp_history = plant.state_to_keypoints(x_history)
    
    plt.figure(figsize=(8,8))
    colormap0 = get_cmap('winter')
    colormap1 = get_cmap('hot')
    
    for t in range(cfg.trj.T):
        plt.plot(x_trj[t,0], x_trj[t,1], marker='o', color=colormap0(
            t / (cfg.trj.T + 1)
        ))
        x_history[t + 1] = plant.dynamics(x_history[t], u_trj[t])
        x_kp_history[t + 1] = plant.state_to_keypoints(x_history[t + 1: t + 2])
        plt.plot(x_kp_history[t,0].cpu().detach().numpy(), x_kp_history[t,1].cpu().detach().numpy(), marker='o', color=colormap1(
            t / (cfg.trj.T + 1)
        ))

    theta = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(theta) * cfg.pendulum_length, np.sin(theta) * cfg.pendulum_length, linestyle="--")
    plt.axis("equal")
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.savefig("closed_loop.png")
    plt.close()


if __name__ == "__main__":
    main()
