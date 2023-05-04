import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from torch.utils.data import TensorDataset

import hydra
import pickle
from omegaconf import DictConfig
import gym, d4rl

from score_po.score_matching import ScoreEstimator
from score_po.nn import MLP, TrainParams, Normalizer
from score_po.dynamical_system import NNDynamicalSystem
from score_po.policy import NNPolicy
from score_po.costs import NNCost
from score_po.policy_optimizer import DRiskScorePolicyOptimizerParams

from gym_policy import GymPolicyOptimizer


@hydra.main(config_path="./config", config_name="policy_optimization")
def main(cfg: DictConfig):
    env = gym.make(cfg.env_name)
    dim_x = env.observation_space.shape[0]
    dim_u = env.action_space.shape[0]
    
    # Load learned dynamnics.
    network_dyn = MLP(dim_x + dim_u, dim_x, cfg.nn_layers)
    system = NNDynamicalSystem(dim_x, dim_u, network_dyn)
    system.load_state_dict(torch.load(cfg.dynamics_weights))
    
    # Load learned costs.
    network_cost = MLP(dim_x + dim_u, 1, cfg.nn_layers)
    cost = NNCost(dim_x, dim_u, network_cost)
    cost.load_state_dict(torch.load(cfg.cost_weights))
    
    # Load learned score function.
    network_sf = MLP(dim_x + dim_u, dim_x + dim_u, cfg.nn_layers)
    sf = ScoreEstimator(dim_x, dim_u, network_sf)
    sf.load_state_dict(torch.load(cfg.score_weights))
    
    # Define policy.
    network_policy = MLP(dim_x, dim_u, cfg.nn_layers_policy)
    policy = NNPolicy(dim_x, dim_u, network=network_policy)

    # Load dynamics.    
    policy_params = DRiskScorePolicyOptimizerParams()
    policy_params.load_from_config(cfg)
    policy_params.policy = policy
    policy_params.cost = cost
    policy_params.dynamical_system = system
    policy_params.sf = sf
    policy_params.to_device(cfg.policy.device)
    
    # Define optimizer
    optimizer = GymPolicyOptimizer(policy_params, cfg.env_name)
    optimizer.iterate()


if __name__ == "__main__":
    main()
