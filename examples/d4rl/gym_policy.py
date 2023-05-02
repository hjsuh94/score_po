import torch
import numpy as np
from tqdm import tqdm

import gym, d4rl

from score_po.policy import Policy, NNPolicy
from score_po.nn import MLP, TrainParams, Normalizer
from score_po.policy_optimizer import (
    DRiskScorePolicyOptimizer,
    DRiskScorePolicyOptimizerParams,
)


class GymPolicyOptimizer(DRiskScorePolicyOptimizer):
    def __init__(
        self, params: DRiskScorePolicyOptimizerParams, env_name: str, **kwargs
    ):
        super().__init__(params=params, **kwargs)
        self.env = gym.make(env_name)

    def sample_initial_state_batch(self):
        initial_state_batch = np.zeros((self.params.batch_size, self.ds.dim_x))
        for b in range(self.params.batch_size):
            initial_state_batch[b] = self.env.reset()
        return torch.Tensor(initial_state_batch).to(self.params.device)


class GymPolicyEvaluator:
    """Given environment and policy, evaluate the performance."""

    def __init__(self, env_name: str, policy: Policy):
        self.env = gym.make(env_name)
        self.dim_x = self.env.observation_space.shape[0]
        self.dim_u = self.env.action_space.shape[0]
        self.policy = policy
        # Taken from D4RL paper, page 7, evaluation protocol.
        self.batch_size = 100

    def check_input_consistency(self):
        assert self.policy.dim_x == self.dim_x
        assert self.policy.dim_u == self.dim_u

    def get_policy_score(self):
        obs = self.env.reset()
        done = False
        time = 0
        returns = 0.0
        while not done:
            action = self.policy(torch.Tensor(obs)[None, :], time)[0].detach().numpy()
            obs, reward, done, _ = self.env.step(action)

            time += 1
            returns += reward
        return self.env.get_normalized_score(returns)

    def get_policy_score_mean(self):
        policy_scores = np.zeros(self.batch_size)
        for b in tqdm(range(self.batch_size)):
            policy_scores[b] = self.get_policy_score()
        return np.mean(policy_scores)
