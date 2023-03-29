# %% [markdown]
# # Training the Score Function Estimator
#

# %%
import numpy as np
import torch
import matplotlib.pyplot as plt

from score_po.policy_optimizer import (
    PolicyOptimizerParams,
    FirstOrderPolicyDRiskOptimizer)
from score_po.dynamical_system import DynamicalSystem
from score_po.costs import QuadraticCost
from score_po.policy import NNPolicy, TimeVaryingOpenLoopPolicy
from score_po.nn import MLP
from score_po.score_matching import ScoreFunctionEstimator

# 1. Set up parameters.
params = PolicyOptimizerParams()
params.T = 20
params.x0_upper = torch.Tensor([0.45, 0.1])
params.x0_lower = torch.Tensor([0.45, 0.1])
params.batch_size = 1
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
xd = torch.tensor([-0.2, 0.0])
cost = QuadraticCost(Q, R, Qd, xd)
params.cost = cost

# 3. Set up policy and initial guess.
policy = TimeVaryingOpenLoopPolicy(2, 2, params.T)
params.policy = policy
params.policy_params_0 = policy.get_parameters()

# 4. Set up the score function estimator
score_network = MLP(5, 4, [128, 128])
sf = ScoreFunctionEstimator(score_network, 2, 2)
sf.load_network_parameters("examples/light_dark/nn_weights_ds.pth")

# debug.
# 4. Run the optimizer.
beta_lst = [1.0, 10.0, 30.0, 50.0, 100.0]
color_lst = ["g", "b", "m", "y", "r"]

plt.figure()
circle = plt.Circle((90 / 128 - 0.5, 64 / 128 - 0.5), 24 / 128, color='r',
                    fill=None, linestyle='--')
plt.gca().add_patch(circle)
    
for idx, beta in enumerate(beta_lst):
    optimizer = FirstOrderPolicyDRiskOptimizer(params=params, sf=sf, beta=beta)
    optimizer.iterate()

    # 5. Run the policy.
    x0_batch = optimizer.sample_initial_state_batch()
    zero_noise_trj = torch.zeros(params.batch_size, params.T, 2)
    x_trj, u_trj = optimizer.rollout_policy_batch(x0_batch, zero_noise_trj)
    x_trj = x_trj.detach().numpy()

    for b in range(params.batch_size):
        plt.plot(x_trj[b,:,0], x_trj[b,:,1], color_lst[idx], label=str(beta))

plt.plot(-0.2, 0.0, label='goal', marker='x', markersize=20)
plt.plot(0.45, 0.1, label='start', marker='x', markersize=20)
        
plt.legend()
plt.xlim([-0.5, 0.5])
plt.ylim([-0.5, 0.5])
plt.savefig("results.png")
plt.close()
