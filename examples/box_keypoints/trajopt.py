import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from torch.utils.data import TensorDataset
import os, copy

import hydra
from hydra.utils import get_original_cwd
import pickle
from omegaconf import DictConfig

from score_po.score_matching import ScoreEstimator
from score_po.nn import MLP, TrainParams
from score_po.policy_optimizer import (
    PolicyOptimizer,
    PolicyOptimizerParams,
    DRiskScorePolicyOptimizer,
    DRiskScorePolicyOptimizerParams,
)
from score_po.costs import QuadraticCost
from score_po.policy import TimeVaryingOpenLoopPolicy, TimeVaryingStateFeedbackPolicy
from box_pushing_system import (
    PlanarPusherSystem,
    KeypointPusherSystem,
    TorchPlanarPusherSystem,
)


@hydra.main(config_path="./config", config_name="trajopt")
def main(cfg: DictConfig):
    # Load learned dynamics.
    network = MLP(14, 12, cfg.nn_layers_dyn)
    system = KeypointPusherSystem(network)
    true_system = PlanarPusherSystem()
    true_system_torch = TorchPlanarPusherSystem()
    system.load_state_dict(
        torch.load(
            os.path.join(
                get_original_cwd(),
                "examples/box_keypoints/weights/checkpoint_dynamics_augment.pth",
            )
        )
    )

    # Load learned score function.
    network = MLP(14, 14, cfg.nn_layers_sf)
    sf = ScoreEstimator(dim_x=12, dim_u=2, network=network)
    sf.load_state_dict(
        torch.load(
            os.path.join(
                get_original_cwd(),
                "examples/box_keypoints/weights/checkpoint_score_hole_augment.pth",
            )
        )
    )

    # Define initial and final state
    x0_pose_np = np.array([-0.3, 0.0, 0, -0.35, 0.0])
    x0_torch = true_system.pose_to_keypoints_x(x0_pose_np)
    print(x0_torch[0:10].reshape(2, 5))

    xd_pose_np = np.array([0.0, 0.3, np.pi/2, 0.0, 0.25])
    xd_torch = true_system.pose_to_keypoints_x(xd_pose_np)

    a = xd_torch[0:10].reshape(2, 5)
    plt.figure()
    plt.plot(a[0, :], a[1, :], "ro")
    plt.savefig("test.png")
    plt.close()

    # Define cost
    Q = torch.diag(torch.concat((torch.ones(10), 0.01 * torch.ones(2))))
    R = 0.01 * torch.diag(torch.ones(2))
    Qd = 10 * Q
    cost = QuadraticCost(Q=Q, R=R, Qd=Qd, xd=xd_torch)

    policy_params = DRiskScorePolicyOptimizerParams()
    policy_params.load_from_config(cfg)
    policy_params.x0_upper = x0_torch
    policy_params.x0_lower = x0_torch
    policy_params.policy = TimeVaryingStateFeedbackPolicy(dim_x=12, dim_u=2, T=cfg.policy.T)
    policy_params.cost = cost
    policy_params.dynamical_system = system
    policy_params.beta = cfg.policy.beta
    policy_params.sf = sf

    # Define optimizer
    optimizer = DRiskScorePolicyOptimizer(policy_params)

    # Define true optimizer

    true_params = copy.copy(policy_params)
    true_params.dynamical_system = true_system_torch
    true_optimizer = PolicyOptimizer(true_params)

    # optimizer.iterate(true_opt=true_optimizer)
    optimizer.iterate(true_opt=true_optimizer)


if __name__ == "__main__":
    main()
