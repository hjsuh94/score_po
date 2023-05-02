import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset

import hydra
from omegaconf import DictConfig

from score_po.score_matching import ScoreEstimatorXu
from score_po.nn import MLP, TrainParams, Normalizer

from examples.light_dark.dynamics import SingleIntegrator
from examples.light_dark.environment import Environment

@hydra.main(config_path="../config", config_name="train")
def main(cfg: DictConfig):
    env = Environment()
    env.add_ellipse([0, 0], 0.4, 0.4)
    x_batch = env.sample_points(cfg.dataset_size)
    u_batch = 0.1 * 2.0 * (torch.rand(x_batch.shape[0], 2) - 0.5)
    u_normalizer = Normalizer(k =0.1 * torch.ones(2), b= torch.zeros(2))    
    dataset = TensorDataset(x_batch, u_batch)
    
    params = TrainParams()
    params.load_from_config(cfg)
    
    network = MLP(4, 4, cfg.nn_layers)
    sf = ScoreEstimatorXu(2, 2, network, u_normalizer=u_normalizer)
    sf.to(cfg.train.device)
    sf.train_network(dataset, params, sigma=0.1 * torch.ones(1))
    sf.to("cpu")

    plt.figure()
    plt.subplot(1,2,1)    
    # plot the gradients.
    X, Y = np.meshgrid(range(32), range(32))
    pos = np.vstack([X.ravel(), Y.ravel()]).T
    pos = 2.0 * (torch.Tensor(pos) / 32 - 0.5)

    grads = sf.get_score_z_given_z(
        sf.get_z_from_xu(pos, torch.zeros(pos.shape[0],2)))[:,:2]
    grads = grads.detach().numpy()
    UV = np.swapaxes(grads, 0, 1).reshape(2, 32, 32)
    plt.quiver(X, Y, UV[0, :, :], UV[1, :, :])
    
    plt.subplot(1,2,2)
    grads = sf.get_score_z_given_z(
        sf.get_z_from_xu(torch.zeros(pos.shape[0],2), pos))[:,2:]
    grads = grads.detach().numpy()
    UV = np.swapaxes(grads, 0, 1).reshape(2, 32, 32)
    plt.quiver(X, Y, UV[0, :, :], UV[1, :, :])
    
    plt.savefig("quiver.png")
    plt.close()


if __name__ == "__main__":
    main()