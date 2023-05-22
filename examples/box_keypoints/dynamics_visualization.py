import os
import torch
import numpy as np
import matplotlib.pyplot as plt

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from score_po.nn import MLP
from box_pushing_system import (
    KeypointPusherSystem, PlanarPusherSystem, render_dynamics)

def render_dynamics_compare(system, x_pose, u):
    true_system = PlanarPusherSystem()
    # Compute current keypoints.
    x_keypts = system.dynamics_true.pose_to_keypoints_x(x_pose)
    keypts = x_keypts[0:10].reshape(2, 5)
    pusher = x_keypts[10:12]
    
    # Compute true dynamics.
    x_pose_next = true_system.dynamics(x_pose, u)
    x_keypts_next_true = true_system.pose_to_keypoints_x(x_pose_next)
    keypts_next_true = x_keypts_next_true[0:10].reshape(2,5)
    pusher_next_true = x_keypts_next_true[10:12]
    
    plt.figure()    
    render_dynamics(
        plt.gca(), keypts, pusher, u, 
        keypts_next_true, pusher_next_true, color='blue')
    
    # Compute predicted dynamics.
    x_keypts_next_pred = system.dynamics(x_keypts, torch.Tensor(u)).detach().numpy()
    keypts_next_pred = x_keypts_next_pred[0:10].reshape(2,5)
    pusher_next_pred = x_keypts_next_pred[10:12]
    render_dynamics(
        plt.gca(), keypts, pusher, u, 
        keypts_next_pred, pusher_next_pred, color='red')
    plt.axis('equal')
    plt.savefig("dynamics_test.png")
    plt.close()


@hydra.main(config_path="./config", config_name="dynamics_training")
def main(cfg: DictConfig):
    network = MLP(14, 12, cfg.nn_layers)
    system = KeypointPusherSystem(network)
    system.load_state_dict(
        torch.load(
            os.path.join(
                get_original_cwd(),
                "examples/box_keypoints/weights/checkpoint_dynamics_hole_augment.pth",
            )
        )
    )

    render_dynamics_compare(system, 
        np.array([-0.35, 0.0, 0.0, -0.4, 0.0]), np.array([0.1, 0.0])
    )


if __name__ == "__main__":
    main()
