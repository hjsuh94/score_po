import numpy as np
import torch
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig

from score_po.policy_optimizer import PolicyOptimizerParams, PolicyOptimizer
from score_po.dynamical_system import DynamicalSystem
from score_po.costs import QuadraticCost
from score_po.policy import TimeVaryingStateFeedbackPolicy
from score_po.nn import MLP

from examples.light_dark.dynamics import SingleIntegrator

@hydra.main(config_path="../config", config_name="trajopt")
def main(cfg: DictConfig):
    # 1. Set up parameters.
    params = PolicyOptimizerParams()    
    
    # 2. Load dynamics.
    dynamics = SingleIntegrator()
    params.dynamical_system = dynamics
    
    # 3. Load costs.
    cost = QuadraticCost()
    cost.load_from_config(cfg)
    params.cost = cost
    
    # 4. Set up policy and initial guess.
    policy = TimeVaryingStateFeedbackPolicy(2, 2, cfg.policy.T)
    params.policy = policy
    
    # 5. Load rest of the parameters.
    params.load_from_config(cfg)

    # 6. Run the optimizer.
    optimizer = PolicyOptimizer(params)
    optimizer.iterate()

    # 7. Run the policy.
    x0_batch = optimizer.sample_initial_state_batch().to(
        cfg.policy.device)
    zero_noise_trj = torch.zeros(params.batch_size, params.T, 2).to(
        cfg.policy.device)
    x_trj, u_trj = optimizer.rollout_policy_batch(x0_batch, zero_noise_trj)
    x_trj = x_trj.detach().cpu().numpy()

    plt.figure()
    for b in range(params.batch_size):
        plt.plot(x_trj[b, :, 0], x_trj[b, :, 1], "springgreen")
    plt.savefig("trajopt_feedback.png")
    plt.close()

if __name__ == "__main__":
    main()
