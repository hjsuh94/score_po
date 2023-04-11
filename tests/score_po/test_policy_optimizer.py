import numpy as np
import pytest
import torch
import torch.nn as nn

import score_po.nn as mut
from typing import Literal
from hydra import compose, initialize

from score_po.costs import QuadraticCost
import score_po.policy_optimizer as mut
from score_po.policy import (
    TimeVaryingOpenLoopPolicy,
    TimeVaryingStateFeedbackPolicy,
    NNPolicy)
from score_po.dynamical_system import DynamicalSystem
from score_po.nn import MLP

class TestPolicyConfig:
    def test_cfg_load(self):
        params = mut.PolicyOptimizerParams()
        with initialize(config_path="./config"):
            cfg = compose(config_name="policy_params")
            params.load_from_config(cfg)

        np.testing.assert_equal(params.T, 20)
        np.testing.assert_allclose(params.x0_upper, torch.Tensor([0.3, 0.2]))
        np.testing.assert_allclose(params.x0_lower, torch.Tensor([0.28, 0.18]))
        np.testing.assert_equal(params.batch_size, 1)
        np.testing.assert_allclose(params.std, torch.Tensor([1e-2, 1e-2]))

        np.testing.assert_equal(params.wandb_params.enabled, True)
        np.testing.assert_equal(params.device, "cuda")
        
class SingleIntegrator(DynamicalSystem):
    def __init__(self):
        super().__init__(2, 2)
        self.is_differentiable = True

    def dynamics(self, x, u):
        return x + u

    def dynamics_batch(self, x_batch, u_batch):
        return x_batch + u_batch        

class TestFirstOrderPolicyOptimizerKnownDynamics:
    def initialize_problem(self, device, policy, cfg):

        costs = QuadraticCost()
        costs.load_from_config(cfg)
            
        params = mut.PolicyOptimizerParams()
        params.cost = costs
        params.dynamical_system = SingleIntegrator()
        params.policy = policy
        params.load_from_config(cfg)
        params.device = device # overwrite device for testing
        return params

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_open_loop_policy(self, device):
        with initialize(config_path="./config"):
            self.cfg = compose(config_name="policy_params")
                            
        policy = TimeVaryingOpenLoopPolicy(2, 2, self.cfg.policy.T)
        params = self.initialize_problem(device, policy, self.cfg)
        params.policy_params_0 = policy.get_parameters()
        
        optimizer = mut.FirstOrderPolicyOptimizer(params)
        optimizer.iterate()

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_state_feedback_policy(self, device):
        with initialize(config_path="./config"):
            self.cfg = compose(config_name="policy_params")
                    
        policy = TimeVaryingStateFeedbackPolicy(2, 2, self.cfg.policy.T)
        params = self.initialize_problem(device, policy, self.cfg)        
        params.policy_params_0 = policy.get_parameters()
        
        optimizer = mut.FirstOrderPolicyOptimizer(params)
        optimizer.iterate()
        
    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_nn_policy(self, device):
        with initialize(config_path="./config"):
            self.cfg = compose(config_name="policy_params")
                    
        network = MLP(2, 2, [128, 128])
        policy = NNPolicy(2, 2, network)
        params = self.initialize_problem(device, policy, self.cfg)
        params.policy_params_0 = policy.get_parameters()
        
        optimizer = mut.FirstOrderNNPolicyOptimizer(params)
        optimizer.iterate()
