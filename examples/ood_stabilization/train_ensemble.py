import numpy as np
import torch
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt

from score_po.nn import MLP, EnsembleNetwork, AdamOptimizerParams

# %%
# 1. Insert network.

def target_function(xy_batch):
    return 0.0

params = AdamOptimizerParams()
network_lst = []
for i in range(10):
    net = MLP(2,1, [128, 128, 128])
    net.train()

    params.iters = 1000
    params.batch_size = 64
    optimizer = optim.Adam(net.parameters(), 1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, params.iters)
    for epoch in tqdm(range(params.iters)):
        r_batch = torch.rand(params.batch_size)[:,None]
        theta_batch = 2.0 * torch.pi * torch.rand(params.batch_size)[:,None]
        xy_batch = torch.cat((r_batch * torch.cos(theta_batch), 
                            r_batch * torch.sin(theta_batch)), dim=1)
        
        loss = ((net(xy_batch) - target_function(xy_batch)) ** 2.0).mean(dim=0)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
    network_lst.append(net)

ensemble = EnsembleNetwork(2, 1, network_lst)
ensemble.save_ensemble("ensemble")