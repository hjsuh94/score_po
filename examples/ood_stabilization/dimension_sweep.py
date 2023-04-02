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

def target_function(xy_batch):
    return 0.0

def generate_data(dim, batch_size):
    r_batch = torch.rand(batch_size)[:,None]
    # sample random vectors.
    v_batch = torch.normal(0.0, 1.0, size=(batch_size, dim))
    v_batch = v_batch / v_batch.norm(dim=1)[:,None]
    xy_batch = r_batch * v_batch
    return xy_batch

def train_ensemble_dim(dim, ensemble_size):
    params = AdamOptimizerParams()    
    params.iters = 1000
    params.batch_size = 64
    
    network_lst = []
    for i in range(ensemble_size):
        net = MLP(dim, 1, [128, 128, 128])
        net.train()
        optimizer = optim.Adam(net.parameters(), 1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, params.iters)
        for epoch in tqdm(range(params.iters)):
            xy_batch = generate_data(dim, params.batch_size)
            loss = ((net(xy_batch) - target_function(xy_batch)) ** 2.0).mean(dim=0)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
        network_lst.append(net)

    ensemble = EnsembleNetwork(dim, 1, network_lst)
    return ensemble

def train_sf_dim(dim):
    params = AdamOptimizerParams()
    params.iters = 1000
    params.batch_size = 64

    network = MLP(dim+1, dim, [128, 128, 128])
    sf = ScoreFunctionEstimator(network, dim, 0)
    optimizer = optim.Adam(network.parameters(), 1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, params.iters)

    for epoch in tqdm(range(params.iters)):
        xy_batch = generate_data(dim, params.batch_size)
        optimizer.zero_grad()
        loss = sf.evaluate_denoising_loss_with_sigma(xy_batch, 0.05)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
    return sf

def generate_initial(batch_size, dim, window):
    return 2.0 * window * torch.rand(batch_size, dim) - window

def plot_rollouts(rollouts, name):
    plt.figure()
    circle = plt.Circle((0,0), 1.0, color='r', fill=False)
    plt.gca().add_patch(circle)
    for i in range(rollouts.shape[1]):
        plt.plot(rollouts[:,i,0], rollouts[:,i,1], 'b-')
    plt.plot(rollouts[-1,:,0], rollouts[-1,:,1], color='springgreen',
             marker='o', linestyle='', markersize=1.0)
    plt.savefig(name)
    plt.close()

def evaluate_batch(x_batch):
    """
    Given x_batch of shape (B, dim_x), compute some metrics.
    """
    # Check how many of them went out of distribution.
    ood_index = (x_batch ** 2).sum(dim=1) > 1
    # Chamfer distance.
    ood_samples = x_batch[ood_index,:]
    if ood_samples.shape[0] == 0:
        return torch.zeros(1), 0
    else:
        chamfer = (ood_samples ** 2).sum(dim=1).mean(dim=0)
        return chamfer, ood_samples.shape[0]
    
dims = [2, 5, 10, 100, 1000]
ensemble_sizes = [2, 5, 10, 20]
num_tries = 10

chamfer_storage = np.zeros((len(dims), len(ensemble_sizes), num_tries, 2))
ood_storage = np.zeros((len(dims), len(ensemble_sizes), num_tries, 2))

for i, dim in enumerate(dims):
    for j, ensemble_size in enumerate(ensemble_sizes):
        for k in range(num_tries):
            ensemble = train_ensemble_dim(dim, ensemble_size)
            sf = train_sf_dim(dim)
            x_batch = generate_initial(100, dim, 10.0)
            
            rollouts_ensemble = langevin(x_batch,
                lambda x: ensemble.get_var_gradients(x),
                1e-2, 1000)
            
            rollouts_sf = langevin(x_batch, 
                lambda x: -sf.get_score_z_given_z(x, 0.05),
                1e-2, 1000)
            
            plot_rollouts(rollouts_ensemble.detach().numpy(),
                          "figures_ensemble/dim{:04d}_size{:03d}_trial{:02d}.png".format(
                              dim, ensemble_size, k
                          ))
            
            plot_rollouts(rollouts_sf.detach().numpy(),
                          "figures_sf/dim{:04d}_size{:03d}_trial{:02d}.png".format(
                              dim, ensemble_size, k
                          ))            
            
            chamfer, ood = evaluate_batch(rollouts_ensemble[-1])
            chamfer_storage[i,j,k,0] = chamfer.detach().numpy()
            ood_storage[i,j,k,0] = ood
            
            chamfer, ood = evaluate_batch(rollouts_sf[-1])
            chamfer_storage[i,j,k,1] = chamfer.detach().numpy()
            ood_storage[i,j,k,1] = ood
            
            
np.save("chamfer_storage.npy", chamfer_storage)
np.save("ood_storage.npy", ood_storage)