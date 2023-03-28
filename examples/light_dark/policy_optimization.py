# %% [markdown]
# # Training the Score Function Estimator
#

# %%
import numpy as np
import torch
import matplotlib.pyplot as plt

from score_po.optimizer import PolicyOptimizerParams, FirstOrderNNPolicyOptimizer
from score_po.dynamical_system import DynamicalSystem
from score_po.costs import QuadraticCost
from score_po.policy import NNPolicy
from score_po.nn_architectures import MLP

# 1. Set up parameters.
params = PolicyOptimizerParams()
params.T = 10
params.x0_upper = torch.Tensor([0.5, 0.5])
params.x0_lower = -torch.Tensor([0.5, 0.5])
params.batch_size = 64
params.std = 1e-2
params.lr = 1e-3
params.max_iters = 1000

# 1. Set up dynamical system.
class SingleIntegrator(DynamicalSystem):
    def __init__(self):
        super().__init__(2, 2)
        self.is_differentiable = True

    def dynamics(self, x, u):
        return x + u

    def dynamics_batch(self, x_batch, u_batch):
        return x_batch + u_batch


dynamics = SingleIntegrator()
params.dynamical_system = dynamics

# 2. Set up cost.
Q = torch.eye(2)
R = torch.eye(2)
Qd = params.T * torch.eye(2)
xd = torch.zeros(2)
cost = QuadraticCost(Q, R, Qd, xd)
params.cost = cost

# 3. Set up policy and initial guess.
network = MLP(2, 2, [64, 64, 64, 64])
policy = NNPolicy(2, 2, network)
params.policy = policy
params.policy_params_0 = policy.get_parameters()

# debug.
# 4. Run the optimizer.
optimizer = FirstOrderNNPolicyOptimizer(params)
x0_batch = torch.rand(params.batch_size, 2, requires_grad=False)
zero_noise_trj = torch.zeros(params.batch_size, params.T, 2)

# --------------------------------------------------
torch.autograd.set_detect_anomaly(True)
optimizer.policy.net.train()
optimizer.policy.net.zero_grad()
u0_batch = optimizer.policy.get_action_batch(x0_batch, 0)
loss = (u0_batch**2).sum(dim=-1).mean(dim=0)
loss.backward()
print(optimizer.policy.net.get_vectorized_gradients())
print("====Can backprop through policy evaluation.")

# --------------------------------------------------
optimizer.policy.net.train()
optimizer.policy.net.zero_grad()
u0_batch = optimizer.policy.get_action_batch(x0_batch, 0)
x1_batch = optimizer.ds.dynamics_batch(x0_batch, u0_batch)
loss = (x1_batch**2).sum(dim=-1).mean(dim=0)
loss.backward()
print(optimizer.policy.net.get_vectorized_gradients())
print("====Can backprop through first dynamics evaluation.")

# --------------------------------------------------
optimizer.policy.net.train()
optimizer.policy.net.zero_grad()
u0_batch = optimizer.policy.get_action_batch(x0_batch, 0)
x1_batch = optimizer.ds.dynamics_batch(x0_batch, u0_batch)
u1_batch = optimizer.policy.get_action_batch(x1_batch, 1)
x2_batch = optimizer.ds.dynamics_batch(x1_batch, u1_batch)
u2_batch = optimizer.policy.get_action_batch(x2_batch, 1)
x3_batch = optimizer.ds.dynamics_batch(x2_batch, u2_batch)
loss = (x2_batch**2).sum(dim=-1).mean(dim=0)
loss.backward()
print(optimizer.policy.net.get_vectorized_gradients())
print("====Can backprop through chained dynamics evaluation.")

# --------------------------------------------------
optimizer.policy.net.train()
optimizer.policy.net.zero_grad()
x_trj, u_trj = optimizer.rollout_policy_batch(x0_batch, zero_noise_trj)
loss = (u_trj**2).sum(dim=-1).mean(dim=-1).mean(dim=0)
loss.backward()
print(optimizer.policy.net.get_vectorized_gradients())
print("====Can backprop through multiple dynamics evaluation.")
