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
params.lr = 1e-6
params.max_iters = 100

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
R = 1e-1 * torch.eye(2)
Qd = 100 * params.T * torch.eye(2)
xd = torch.zeros(2)
cost = QuadraticCost(Q, R, Qd, xd)
params.cost = cost

# 3. Set up policy and initial guess.
network = MLP(2, 2, [128, 128])
policy = NNPolicy(2, 2, network)
params.policy = policy
params.policy_params_0 = policy.get_parameters()

# debug.
# 4. Run the optimizer.
optimizer = FirstOrderNNPolicyOptimizer(params)
optimizer.iterate()

# 5. Run the policy.
x0_batch = optimizer.sample_initial_state_batch()
zero_noise_trj = torch.zeros(params.batch_size, params.T, 2)
x_trj, u_trj = optimizer.rollout_policy_batch(x0_batch, zero_noise_trj)
x_trj = x_trj.detach().numpy()

plt.figure()
for b in range(params.batch_size):
    plt.plot(x_trj[b,:,0], x_trj[b,:,1])
plt.savefig("results.png")
plt.close()
