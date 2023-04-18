import os
import torch
import numpy as np
import matplotlib.pyplot as plt

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from score_po.nn import MLP
from box_pushing_system import KeypointPusherSystem


@hydra.main(config_path="./config", config_name="dynamics_training")
def main(cfg: DictConfig):
    network = MLP(14, 12, cfg.nn_layers)
    system = KeypointPusherSystem(network, project_dir=get_original_cwd())
    system.load_network_parameters(
        os.path.join(
            get_original_cwd(), "examples/box_keypoints/weights/checkpoint_dynamics.pth"
        )
    )

    system.render_dynamics_test(np.array([0, 0, 0, 0.1, 0.0]), np.array([-0.1, 0.05]))


if __name__ == "__main__":
    main()
