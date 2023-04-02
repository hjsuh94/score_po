import numpy as np
import torch
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt

from score_po.score_matching import ScoreFunctionEstimator
from score_po.nn import MLP, EnsembleNetwork, AdamOptimizerParams
from langevin import langevin


# %%
# 1. Insert network.

network_lst = []
for i in range(10):
    net = MLP(2,1, [128, 128, 128])
    network_lst.append(net)

ensemble = EnsembleNetwork(2, 1, network_lst)
ensemble.load_ensemble("ensemble")

#%% 

"""
x_space = torch.linspace(-3, 3, 100)
y_space = torch.linspace(-3, 3, 100)
X, Y = torch.meshgrid(x_space, y_space)
coords = torch.vstack((X.ravel(), Y.ravel())).T

mean = ensemble(coords)
var = ensemble.get_var(coords)

mean_Z = mean.view(100, 100)
var_Z = var.view(100, 100)

grad_Z = ensemble.get_var_gradients(coords)
UV = -torch.swapaxes(grad_Z, 0,1).reshape(2, 100, 100)

plt.figure()
plt.subplot(1,2,1)
plt.pcolormesh(X.numpy(), Y.numpy(), mean_Z.detach().numpy())
plt.subplot(1,2,2)
plt.pcolormesh(X.numpy(), Y.numpy(), var_Z.detach().numpy())
plt.savefig("results.png")
plt.close()
"""

langevin_iters = 1000
grid_size = 16
window = 2.0
plt.figure()
circle = plt.Circle((0,0), 1.0, color='r', fill=False)
x_space = torch.linspace(-window, window, grid_size)
y_space = torch.linspace(-window, window, grid_size)
X, Y = torch.meshgrid(x_space, y_space)
coords = torch.vstack((X.ravel(), Y.ravel())).T
grad_Z = ensemble.get_var_gradients(coords)
UV = -torch.swapaxes(grad_Z, 0,1).reshape(2, grid_size, grid_size)

plt.subplot(1,2,1)
plt.quiver(X, Y, UV[0,:,:], UV[1,:,:])
plt.gca().add_patch(circle)

x0_batch = 2.0 * window * torch.rand(100,2) - window
history = langevin(x0_batch, lambda x: ensemble.get_var_gradients(x), 1e-0,
                   langevin_iters)
for i in range(100):
    plt.plot(history[:,i,0], history[:,i,1], 'b-')
final_iters = history[-1,:,:]
plt.plot(final_iters[:,0], final_iters[:,1],
         color='springgreen', marker='o', linestyle='',
         markersize=1.0)

net = MLP(3, 2, [128, 128, 128])
sf = ScoreFunctionEstimator(net, 2, 0)
sf.load_network_parameters("score.pth")
print(coords.shape)

grad_score = sf.get_score_z_given_z(coords, 0.05)
UV = torch.swapaxes(grad_score, 0, 1).reshape(2, grid_size, grid_size)
UV = UV.detach().numpy()
ax = plt.subplot(1,2,2)
plt.quiver(X, Y, UV[0,:,:], UV[1,:,:])
circle = plt.Circle((0,0), 1.0, color='r', fill=False)
ax.add_patch(circle)

x0_batch = 2.0 * window * torch.rand(100,2) - window
history = langevin(x0_batch, lambda x: -sf.get_score_z_given_z(x, 0.05), 1e-3,
                   langevin_iters)
for i in range(100):
    plt.plot(history[:,i,0].detach().numpy(), history[:,i,1].detach().numpy(), 'b-')
final_iters = history[-1,:,:].detach().numpy()
plt.plot(final_iters[:,0], final_iters[:,1],
         color='springgreen', marker='o', linestyle='',
         markersize=1.0)

plt.savefig("grad.png")
plt.close()


# %%
 