import os
import torch
import numpy as np
import matplotlib.pyplot as plt

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from score_po.nn import MLP
from score_po.score_matching import ScoreEstimator, langevin_dynamics
from box_pushing_system import KeypointPusherSystem


@hydra.main(config_path="./config", config_name="score_training")
def main(cfg: DictConfig):
    network = MLP(14, 14, cfg.nn_layers)
    sf = ScoreEstimator(dim_x=12, dim_u=2, network=network)
    sf.net.load_state_dict(
        torch.load(
        os.path.join(
            get_original_cwd(), 
            "examples/box_keypoints/weights/checkpoint_score.pth"
        )))
    
    n_steps = 1000
    x0 = 0.8 * (torch.rand(14) - 0.5)
    x_history = langevin_dynamics(x0, sf, 1e-3, n_steps, noise=False)
    keypts_history = x_history[:,:10].reshape(n_steps, 2,5).detach().numpy()
    
    plt.figure()
    for i in range(5):
        plt.plot(keypts_history[:,0,i], keypts_history[:,1,i], 'ro')
    plt.savefig("test.png")
    plt.close()

if __name__ == "__main__":
    main()
