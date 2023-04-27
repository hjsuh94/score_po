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
    NNPolicy,
)
from score_po.dynamical_system import DynamicalSystem, NNDynamicalSystem
from score_po.nn import MLP
from score_po.score_matching import ScoreEstimatorXu


class TestPolicyConfig:
    def test_cfg_load(self):
        params = mut.PolicyOptimizerParams()
        with initialize(config_path="./config"):
            cfg = compose(config_name="policy_params")
            params.load_from_config(cfg)

        np.testing.assert_equal(params.T, 20)
        np.testing.assert_allclose(params.x0_upper.cpu(), torch.Tensor([0.3, 0.2]))
        np.testing.assert_allclose(params.x0_lower.cpu(), torch.Tensor([0.28, 0.18]))
        np.testing.assert_equal(params.batch_size, 16)
        np.testing.assert_allclose(params.std, 1e-2)
        np.testing.assert_equal(params.first_order, True)

        np.testing.assert_equal(params.wandb_params.enabled, True)
        np.testing.assert_equal(params.device, "cuda")

    def test_drisk_cfg_load(self):
        params = mut.DRiskScorePolicyOptimizerParams()
        with initialize(config_path="./config"):
            cfg = compose(config_name="policy_params")
            params.load_from_config(cfg)

        np.testing.assert_equal(params.T, 20)
        np.testing.assert_allclose(params.x0_upper.cpu(), torch.Tensor([0.3, 0.2]))
        np.testing.assert_allclose(params.x0_lower.cpu(), torch.Tensor([0.28, 0.18]))
        np.testing.assert_equal(params.batch_size, 16)
        np.testing.assert_allclose(params.std, 1e-2)

        np.testing.assert_equal(params.wandb_params.enabled, True)
        np.testing.assert_equal(params.device, "cuda")

        np.testing.assert_equal(params.beta, 0.1)


class SingleIntegrator(DynamicalSystem):
    def __init__(self):
        super().__init__(2, 2)
        self.is_differentiable = True

    def dynamics(self, x, u):
        return x + u

    def dynamics_batch(self, x_batch, u_batch):
        return x_batch + u_batch


class TestPolicyOptimizerKnownDynamics:
    def initialize_problem(self, device, policy, cfg):

        costs = QuadraticCost()
        costs.load_from_config(cfg)

        params = mut.PolicyOptimizerParams()
        params.cost = costs
        params.dynamical_system = SingleIntegrator()
        params.policy = policy
        params.load_from_config(cfg)
        params.to_device(device)
        return params

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    @pytest.mark.parametrize("first_order", (True, False))
    def test_open_loop_policy(self, device, first_order):
        with initialize(config_path="./config"):
            self.cfg = compose(config_name="policy_params")
            self.cfg.policy.device = device
            self.cfg.policy.first_order = first_order

        policy = TimeVaryingOpenLoopPolicy(2, 2, self.cfg.policy.T)
        params = self.initialize_problem(device, policy, self.cfg)

        optimizer = mut.PolicyOptimizer(params)
        optimizer.iterate()

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    @pytest.mark.parametrize("first_order", (True, False))
    def test_state_feedback_policy(self, device, first_order):
        with initialize(config_path="./config"):
            self.cfg = compose(config_name="policy_params")
            self.cfg.policy.device = device
            self.cfg.policy.first_order = first_order

        policy = TimeVaryingStateFeedbackPolicy(2, 2, self.cfg.policy.T)
        params = self.initialize_problem(device, policy, self.cfg)

        optimizer = mut.PolicyOptimizer(params)
        optimizer.iterate()

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    @pytest.mark.parametrize("first_order", (True, False))
    def test_nn_policy(self, device, first_order):
        with initialize(config_path="./config"):
            self.cfg = compose(config_name="policy_params")
            self.cfg.policy.device = device
            self.cfg.policy.first_order = first_order

        network = MLP(2, 2, [128, 128])
        policy = NNPolicy(2, 2, network)
        params = self.initialize_problem(device, policy, self.cfg)

        optimizer = mut.PolicyOptimizer(params)
        optimizer.iterate()

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    @pytest.mark.parametrize("first_order", (True, False))
    def test_torch_optimizer(self, device, first_order):
        with initialize(config_path="./config"):
            self.cfg = compose(config_name="policy_params")
            self.cfg.policy.device = device
            self.cfg.policy.first_order = first_order

        network = MLP(2, 2, [128, 128])
        policy = NNPolicy(2, 2, network)
        params = self.initialize_problem(device, policy, self.cfg)

        params.torch_optimizer = torch.optim.Adadelta
        optimizer = mut.PolicyOptimizer(params)
        optimizer.iterate()

        params.torch_optimizer = torch.optim.RMSprop
        optimizer = mut.PolicyOptimizer(params)
        optimizer.iterate()

        params.torch_optimizer = torch.optim.SGD
        optimizer = mut.PolicyOptimizer(params)
        optimizer.iterate()

        params.torch_optimizer = torch.optim.Adam
        optimizer = mut.PolicyOptimizer(params)
        optimizer.iterate()


class TestPolicyOptimizerNNDynamics:
    def initialize_problem(self, device, policy, cfg):
        costs = QuadraticCost()
        costs.load_from_config(cfg)

        network = MLP(4, 2, [128, 128])
        dynamics = NNDynamicalSystem(2, 2, network)
        dynamics.load_state_dict(torch.load("tests/score_po/weights/dynamics.pth"))

        params = mut.PolicyOptimizerParams()
        params.cost = costs
        params.dynamical_system = dynamics
        params.policy = policy
        params.load_from_config(cfg)
        params.to_device(device)
        return params

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    @pytest.mark.parametrize("first_order", (True, False))
    def test_open_loop_policy(self, device, first_order):
        with initialize(config_path="./config"):
            self.cfg = compose(config_name="policy_params")
            self.cfg.policy.device = device
            self.cfg.policy.first_order = first_order

        policy = TimeVaryingOpenLoopPolicy(2, 2, self.cfg.policy.T)
        params = self.initialize_problem(device, policy, self.cfg)

        optimizer = mut.PolicyOptimizer(params)
        optimizer.iterate()

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    @pytest.mark.parametrize("first_order", (True, False))
    def test_state_feedback_policy(self, device, first_order):
        with initialize(config_path="./config"):
            self.cfg = compose(config_name="policy_params")
            self.cfg.policy.device = device
            self.cfg.policy.first_order = first_order

        policy = TimeVaryingStateFeedbackPolicy(2, 2, self.cfg.policy.T)
        params = self.initialize_problem(device, policy, self.cfg)

        optimizer = mut.PolicyOptimizer(params)
        optimizer.iterate()

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    @pytest.mark.parametrize("first_order", (True, False))
    def test_nn_policy(self, device, first_order):
        with initialize(config_path="./config"):
            self.cfg = compose(config_name="policy_params")
            self.cfg.policy.device = device
            self.cfg.policy.first_order = first_order

        network = MLP(2, 2, [128, 128])
        policy = NNPolicy(2, 2, network)
        params = self.initialize_problem(device, policy, self.cfg)

        optimizer = mut.PolicyOptimizer(params)
        optimizer.iterate()


class TestDRiskScoreOptimizer:
    def initialize_problem(self, device, policy, cfg):
        costs = QuadraticCost()
        costs.load_from_config(cfg)

        network = MLP(4, 2, [128, 128])
        dynamics = NNDynamicalSystem(2, 2, network)
        dynamics.load_state_dict(torch.load("tests/score_po/weights/dynamics.pth"))

        score_network = MLP(4, 4, [128, 128, 128])
        sf = ScoreEstimatorXu(2, 2, score_network)
        sf.load_state_dict(torch.load("tests/score_po/weights/sf_weights.pth"))

        params = mut.DRiskScorePolicyOptimizerParams()
        params.cost = costs
        params.dynamical_system = dynamics
        params.policy = policy
        params.sf = sf
        params.load_from_config(cfg)
        params.to_device(device)
        return params

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    @pytest.mark.parametrize("first_order", (True, False))
    def test_open_loop_policy(self, device, first_order):
        with initialize(config_path="./config"):
            self.cfg = compose(config_name="policy_params")
            self.cfg.policy.device = device
            self.cfg.policy.first_order = first_order

        policy = TimeVaryingOpenLoopPolicy(2, 2, self.cfg.policy.T)
        params = self.initialize_problem(device, policy, self.cfg)

        optimizer = mut.DRiskScorePolicyOptimizer(params)
        optimizer.iterate()

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    @pytest.mark.parametrize("first_order", (True, False))
    def test_state_feedback_policy(self, device, first_order):
        with initialize(config_path="./config"):
            self.cfg = compose(config_name="policy_params")
            self.cfg.policy.device = device
            self.cfg.policy.first_order = first_order

        policy = TimeVaryingStateFeedbackPolicy(2, 2, self.cfg.policy.T)
        params = self.initialize_problem(device, policy, self.cfg)

        optimizer = mut.DRiskScorePolicyOptimizer(params)
        optimizer.iterate()

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    @pytest.mark.parametrize("first_order", (True, False))
    def test_nn_policy(self, device, first_order):
        with initialize(config_path="./config"):
            self.cfg = compose(config_name="policy_params")
            self.cfg.policy.device = device
            self.cfg.policy.first_order = first_order

        network = MLP(2, 2, [128, 128])
        policy = NNPolicy(2, 2, network)
        params = self.initialize_problem(device, policy, self.cfg)

        optimizer = mut.DRiskScorePolicyOptimizer(params)
        optimizer.iterate()

a = TestDRiskScoreOptimizer()
a.test_open_loop_policy("cuda", True)