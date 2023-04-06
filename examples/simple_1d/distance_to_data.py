# Is the data distance really convex?

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import torch.optim as optim
import torch
import torch.nn as nn
from tqdm import tqdm

plt.figure()

# -----------------------------------------------------------------------------
# 1. Draw data.
# -----------------------------------------------------------------------------

N = 3
data = 2.0 * (np.random.rand(N) - 0.5)
plt.subplot(2, 3, 4)
plt.title("Energy to Data")
plt.plot(data, np.zeros(N), "ro")

# Get log likelihood
x_range = 5
x = np.linspace(-x_range, x_range, 10000)

target_fun = lambda x: x**2.0 * np.sin(x)

# -----------------------------------------------------------------------------
# 2. Fit Ensembles
# -----------------------------------------------------------------------------

from score_po.nn import MLP, EnsembleNetwork, AdamOptimizerParams

data_torch = torch.Tensor(data)[:, None]
label = torch.Tensor(target_fun(data))[:, None]

params = AdamOptimizerParams()
network_lst = []
criterion = nn.MSELoss()

for i in tqdm(range(3)):
    net = MLP(1, 1, [64])
    net.train()

    params.iters = 400
    params.batch_size = 1024
    optimizer = optim.Adam(net.parameters(), 1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, params.iters)

    loss_lst = []
    for epoch in tqdm(range(params.iters)):
        optimizer.zero_grad()
        loss = criterion(net(data_torch), label)
        loss_lst.append(loss.clone().detach().numpy())
        loss.backward()
        optimizer.step()
        scheduler.step()

    network_lst.append(net)

    # plt.figure()
    # plt.plot(loss_lst)
    # plt.show()

ensemble = EnsembleNetwork(1, 1, network_lst)

x_tensor = torch.Tensor(x)[:, None]
z = ensemble(x_tensor).detach().numpy()[:, 0]
std = np.sqrt(ensemble.get_var(x_tensor).detach().numpy())

plt.subplot(2, 3, 3)
plt.title("Ensembles")
plt.plot(x, target_fun(x), "k--")
plt.plot(x, ensemble(x_tensor).detach().numpy())
scale = 30.0
plt.fill_between(
    x, z - scale * std**2.0, z + scale * std**2.0, alpha=0.1, color="black"
)
plt.plot(data, target_fun(data), "ro")

plt.subplot(2, 3, 6)
plt.title("Ensemble Variance")
plt.plot(x, scale * std**2.0)
plt.plot(data, np.zeros(N), "ro")
plt.plot(x, np.abs(z - target_fun(x)))

# -----------------------------------------------------------------------------
# 3. Evaluate data to distance
# -----------------------------------------------------------------------------

# Get distribution.
sigma = 0.3
prob = np.zeros(10000)
for i, data_pt in enumerate(data):
    prob += norm.pdf(x, loc=data_pt, scale=sigma)
normalized_prob = prob / N

z = net(x_tensor).detach().numpy()[:, 0]

quadratic = []
for i, data_pt in enumerate(data):
    quadratic.append(((x - data_pt) / sigma) ** 2.0)
quadratic = np.array(quadratic)
min_dist = np.min(quadratic, axis=0)
softmin_dist = -logsumexp(-quadratic, axis=0)

plt.subplot(2, 3, 1)
plt.title("Single NN + Energy to Data")
plt.plot(x, target_fun(x), "k--")
plt.plot(x, z)
plt.plot(data, target_fun(data), "ro")
scale = 0.1
plt.fill_between(
    x, z - scale * softmin_dist, z + scale * softmin_dist, alpha=0.1, color="black"
)

# Get distance.

plt.subplot(2, 3, 4)
plt.plot(x, scale * softmin_dist)
plt.plot(x, np.abs(z - target_fun(x)))

# -----------------------------------------------------------------------------
# 4. Fit GPs.
# -----------------------------------------------------------------------------


# Gaussian processes training.
kernel = 1.0 * RBF(length_scale=0.1, length_scale_bounds=(1e-1, 10.0))
gp = GaussianProcessRegressor()
gp.fit(data.reshape(-1, 1), target_fun(data).reshape(-1, 1))
z, std = gp.predict(x.reshape(-1, 1), return_std=True)
plt.subplot(2, 3, 2)
plt.title("Gaussian Processes (GP)")
plt.plot(x, target_fun(x), "k--")
plt.plot(data, target_fun(data), "ro")
plt.plot(x, z)
scale = 2.0
plt.fill_between(
    x, z - scale * std**2.0, z + scale * std**2.0, alpha=0.1, color="black"
)

plt.subplot(2, 3, 5)
plt.title("GP Variance")
plt.plot(x, scale * std**2.0)
plt.plot(data, np.zeros(N), "ro")
plt.plot(x, np.abs(z - target_fun(x)))


plt.show()
