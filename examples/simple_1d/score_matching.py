# Does score matching actually give me gradients of the perturbed distribution?

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import torch
from score_po.score_matching import ScoreFunctionEstimator
from score_po.nn import MLP, AdamOptimizerParams
import torch.optim as optim
from tqdm import tqdm

plt.figure()
# -----------------------------------------------------------------------------
# 1. Draw data.
# -----------------------------------------------------------------------------

N = 3
data = 2.0 * (np.random.rand(N) - 0.5)

# Get log likelihood
x_range = 2
x = np.linspace(-x_range, x_range, 10000)

target_fun = lambda x: x**2.0 * np.sin(x)

# -----------------------------------------------------------------------------
# 2. Plot the perturbed distribution.
# -----------------------------------------------------------------------------

# Get distribution.
sigma = 0.01
prob = np.zeros(10000)
for i, data_pt in enumerate(data):
    prob += norm.pdf(x, loc=data_pt, scale=sigma)
normalized_prob = prob / N

plt.subplot(1, 3, 1)
plt.title("Perturbed Data Distribution")
plt.plot(x, normalized_prob)
plt.plot(data, np.zeros(N), "ro")

plt.subplot(1, 3, 2)
plt.title("Log-Likelihood vs. Softmin")
plt.plot(x, -np.log(normalized_prob), label="log-likelihood")
plt.plot(data, np.zeros(N), "ro")


# Get gradients of the distribution.

dists = np.zeros(10000)
grads = np.zeros(10000)

for k in range(10000):
    x_tensor = torch.Tensor([x[k]])
    x_tensor.requires_grad = True

    quad_tensor = torch.zeros(0, 1)
    for i, data_pt in enumerate(data):
        quad_tensor = torch.vstack(
            (quad_tensor, 0.5 * ((x_tensor - data_pt) / sigma) ** 2.0)
        )
    dist = -torch.logsumexp(-quad_tensor, dim=0)
    dist.backward()

    dists[k] = dist.detach().numpy()
    grads[k] = x_tensor.grad.numpy()

plt.plot(x, dists, label="softmin")
plt.legend()

plt.subplot(1, 3, 3)
plt.title("Autodiff vs. Score-Matching")
plt.plot(x, grads, label="Autodiff")
plt.plot(data, np.zeros(N), "ro")

# Score match.

network = MLP(1, 1, [512, 512])
sf = ScoreFunctionEstimator(network, 1, 0)
sf.net.train()

optimizer = optim.Adam(sf.net.parameters(), 1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 10000)


def evaluate_denoising_loss_with_sigma(data, sigma):
    """
    Evaluate denoising loss, Eq.(5) from Song & Ermon.
        data of shape (B, dim_x + dim_u)
        sigma, a scalar variable.
    """
    databar = data + torch.randn_like(data) * sigma
    target = -1 / (sigma**2) * (databar - data)
    scores = sf.net(databar)

    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)

    loss = 0.5 * ((scores - target) ** 2).sum(dim=-1).mean(dim=0)
    return loss


loss_lst = []
for epoch in tqdm(range(10000)):
    optimizer.zero_grad()
    loss = evaluate_denoising_loss_with_sigma(torch.Tensor(data).reshape(-1, 1), sigma)
    loss_lst.append(loss.clone().detach().numpy())
    loss.backward()
    optimizer.step()
    scheduler.step()

# plt.figure()
# plt.plot(loss_lst)
# plt.show()


x_tensor = torch.Tensor(x).reshape(-1, 1)
z = -sf.net(x_tensor).detach().numpy()

plt.plot(x, z, label="Score-Matching")
plt.legend()


plt.show()
