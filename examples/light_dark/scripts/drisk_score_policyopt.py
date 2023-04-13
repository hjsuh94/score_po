import os

import numpy as np
import torch
import matplotlib.pyplot as plt

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from score_po.policy_optimizer import (
    DRiskScorePolicyOptimizerParams,
    DRiskScorePolicyOptimizer,
)
from score_po.dynamical_system import DynamicalSystem
from score_po.costs import QuadraticCost
from score_po.policy import NNPolicy, TimeVaryingOpenLoopPolicy
from score_po.nn import MLP
from score_po.score_matching import ScoreEstimator

from examples.light_dark.dynamics import SingleIntegrator


@hydra.main(config_path="../config", config_name="drisk_policyopt")
def main(cfg: DictConfig):
    # 1. Set up parameters.
    params = DRiskScorePolicyOptimizerParams()

    # 2. Load dynamics.
    dynamics = SingleIntegrator()
    params.dynamical_system = dynamics

    # 3. Load costs.
    cost = QuadraticCost()
    cost.load_from_config(cfg)
    params.cost = cost

    # 4. Set up policy and initial guess.
    network = MLP(2, 2, [128, 128, 128])
    policy = NNPolicy(2, 2, network)
    params.policy = policy

    # 5. Set up score optimizer.
    network = MLP(4, 4, [128, 128, 128])
    sf = ScoreEstimator(2, 2, network)

    sf.load_network_parameters(
        os.path.join(get_original_cwd(), "examples/light_dark/weights/checkpoint.pth")
    )
    params.sf = sf

    # 5. Load rest of the parameters.
    params.load_from_config(cfg)

    # 6. Run the optimizer.
    optimizer = DRiskScorePolicyOptimizer(params)
    optimizer.iterate()

    # 7. Run the policy.
    x0_batch = optimizer.sample_initial_state_batch().to(cfg.policy.device)
    zero_noise_trj = torch.zeros(params.batch_size, params.T, 2).to(cfg.policy.device)
    x_trj, u_trj = optimizer.rollout_policy_batch(x0_batch, zero_noise_trj)
    x_trj = x_trj.detach().cpu().numpy()

    plt.figure()
    for b in range(params.batch_size):
        plt.plot(x_trj[b, :, 0], x_trj[b, :, 1], "springgreen")
    circle = plt.Circle((0, 0), 0.4, fill=False)
    plt.gca().add_patch(circle)

    plt.savefig("drisk_policyopt.png")
    plt.close()


if __name__ == "__main__":
    main()
