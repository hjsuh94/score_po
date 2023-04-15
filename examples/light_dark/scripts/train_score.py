import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset

import hydra
from omegaconf import DictConfig

from score_po.score_matching import ScoreEstimator
from score_po.nn import MLP, TrainParams

from examples.light_dark.dynamics import SingleIntegrator
from examples.light_dark.environment import Environment

@hydra.main(config_path="../config", config_name="train")
def main(cfg: DictConfig):
    env = Environment()
    env.add_ellipse([0, 0], 0.4, 0.4)
    pts = env.sample_points(cfg.dataset_size)
    input_pts = torch.rand(pts.shape[0], 2) - 0.5
    data = torch.hstack((pts, input_pts))
    dataset = TensorDataset(data)
    
    params = TrainParams()
    params.load_from_config(cfg)
    
    network = MLP(4, 4, cfg.nn_layers)
    sf = ScoreEstimator(2, 2, network)
    sf.train_network(dataset, params, 0.1)

    plt.figure()
    plt.subplot(1,2,1)    
    # plot the gradients.
    X, Y = np.meshgrid(range(32), range(32))
    pos = np.vstack([X.ravel(), Y.ravel()]).T
    pos = 2.0 * (torch.Tensor(pos) / 32 - 0.5)

    grads = sf.get_score_x_given_xu(pos, torch.zeros(pos.shape[0],2))
    grads = grads.detach().numpy()
    UV = np.swapaxes(grads, 0, 1).reshape(2, 32, 32)
    plt.quiver(X, Y, UV[0, :, :], UV[1, :, :], scale=5.0)
    
    plt.subplot(1,2,2)
    
    grads = sf.get_score_u_given_xu(torch.zeros(pos.shape[0],2), pos)
    grads = grads.detach().numpy()
    UV = np.swapaxes(grads, 0, 1).reshape(2, 32, 32)
    plt.quiver(X, Y, UV[0, :, :], UV[1, :, :], scale=10.0)
    
    plt.savefig("quiver.png")
    plt.close()


if __name__ == "__main__":
    main()