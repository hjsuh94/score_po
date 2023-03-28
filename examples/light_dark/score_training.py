# %% [markdown]
# # Training the Score Function Estimator
#

# %%
import numpy as np
import torch
import matplotlib.pyplot as plt

from score_po.dataset import Dataset
from score_po.score_matching import ScoreFunctionEstimator
from score_po.nn import MLP, AdamOptimizerParams

from environment import Environment, plot_samples_and_enviroment

env = Environment()
env.generate_ellipse(39, 39, [64, 64])
pts = env.sample_points(500000) / 128 - 0.5

#plt.figure()
#plt.plot(pts[:, 0], pts[:, 1], "ro")
#plt.show()

dataset = Dataset(2, 0)
dataset.add_to_dataset(pts)

# Note that MLP input is 3 because dim_x:2, dim_u:0, sigma:0
network = MLP(3, 2, [64, 64, 64, 64])

params = AdamOptimizerParams()
params.batch_size = 512
params.iters = 1000
params.lr = 1e-3
params.step_lr = 500

sf = ScoreFunctionEstimator(network, 2, 0)
loss_lst = sf.train_network(dataset, params, sigma_max=0.1, sigma_min=0.1, n_sigmas=1)

plt.figure()
plt.plot(loss_lst)
plt.show()
sf.save_network_parameters("examples/light_dark/nn_weights.pth")

#%%
sf = ScoreFunctionEstimator(network, 2, 0)
sf.load_network_parameters("examples/light_dark/nn_weights.pth")

# plot the gradients.
X, Y = np.meshgrid(range(32), range(32))
pos = np.vstack([X.ravel(), Y.ravel()]).T
pos = torch.Tensor(pos) / 32 - 0.5

grads = sf.get_score_z_given_z(pos, 0.1)
grads = grads.detach().numpy()
pos = pos.detach().numpy()

UV = np.swapaxes(grads, 0, 1).reshape(2, 32, 32)

plt.figure()
plt.quiver(X, Y, UV[0, :, :], UV[1, :, :], scale=100.0)
plt.savefig("quiver.png")
plt.close()
