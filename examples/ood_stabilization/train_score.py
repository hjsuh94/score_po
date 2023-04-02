import numpy as np
import torch
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt

from score_po.score_matching import ScoreFunctionEstimator
from score_po.nn import MLP, EnsembleNetwork, AdamOptimizerParams

# %%
# 1. Insert network.

params = AdamOptimizerParams()
params.iters = 1000
params.batch_size = 64

network = MLP(3, 2, [128, 128, 128])
sf = ScoreFunctionEstimator(network, 2, 0)
optimizer = optim.Adam(network.parameters(), 1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, params.iters)

for epoch in tqdm(range(params.iters)):
    r_batch = torch.rand(params.batch_size)[:,None]
    theta_batch = 2.0 * torch.pi * torch.rand(params.batch_size)[:,None]
    xy_batch = torch.cat((r_batch * torch.cos(theta_batch), 
                        r_batch * torch.sin(theta_batch)), dim=1)

    
    optimizer.zero_grad()
    loss = sf.evaluate_denoising_loss_with_sigma(xy_batch, 0.05)
    loss.backward()
    optimizer.step()
    scheduler.step()
    
sf.save_network_parameters("score.pth")
