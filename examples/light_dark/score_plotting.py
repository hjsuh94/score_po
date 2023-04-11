# %% [markdown]
# # Training the Score Function Estimator
#

# %%
import numpy as np
import torch
import matplotlib.pyplot as plt

from score_po.dataset import Dataset
from score_po.score_matching import ScoreEstimator
from score_po.policy_optimizer import OptimizerParams
from score_po.nn import MLP

from environment import Environment, plot_samples_and_enviroment


#%%
network = MLP(3, 2, [64, 64, 64, 64])
sf = ScoreEstimator(network, 2, 0)
sf.load_network_parameters("examples/light_dark/nn_weights.pth")

# plot the gradients.
X, Y = np.meshgrid(range(128), range(128))
pos = np.vstack([X.ravel(), Y.ravel()]).T
pos = torch.Tensor(pos) / 128 - 0.5

grads = sf.get_score_z_given_z(pos, 0.1)
grads = grads.detach().numpy()
pos = pos.detach().numpy()

UV = np.swapaxes(grads, 0, 1).reshape(2, 128, 128)

plt.figure()
plt.quiver(X, Y, UV[0, :, :], UV[1, :, :], scale=400.0)
plt.show()

# %%
