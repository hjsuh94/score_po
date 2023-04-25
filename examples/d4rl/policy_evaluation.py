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

from gym_policy import GymPolicyEvaluator


@hydra.main(config_path="./config", config_name="policy_optimization")
def main(cfg: DictConfig):
    env = gym.make(cfg.env_name)
    dim_x = env.observation_space.shape[0]
    dim_u = env.action_space.shape[0]

    # Define policy.
    network_policy = MLP(dim_x, dim_u, cfg.nn_layers_policy)
    policy = NNPolicy(dim_x, dim_u, network=network_policy)
    policy.load_state_dict(torch.load(
        cfg.policy_weights
    ))

    # Load dynamics.    
    policy_eval = GymPolicyEvaluator(cfg.env_name, policy)    
    score = policy_eval.get_policy_score_mean()
    print(score)

if __name__ == "__main__":
    main()
