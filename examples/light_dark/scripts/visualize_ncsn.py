import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
import os

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from score_po.score_matching import NoiseConditionedScoreEstimatorXu
from score_po.nn import MLPwEmbedding, TrainParams, Normalizer

from examples.light_dark.dynamics import SingleIntegrator
from examples.light_dark.environment import Environment

@hydra.main(config_path="../config", config_name="train")
def main(cfg: DictConfig):
    u_normalizer = Normalizer(k =0.1 * torch.ones(2), b= torch.zeros(2))    
    
    params = TrainParams()
    params.load_from_config(cfg)
    
    
    network = MLPwEmbedding(4, 4, cfg.nn_layers, 10)
    sf = NoiseConditionedScoreEstimatorXu(
        2, 2, network, u_normalizer=u_normalizer)
    sf.load_state_dict(torch.load(
        os.path.join(get_original_cwd(),
                     "examples/light_dark/weights/ncsn_xu.pth")))
    
    print(sf.sigma_lst)


    # plot the gradients.
    X, Y = np.meshgrid(range(32), range(32))
    pos = np.vstack([X.ravel(), Y.ravel()]).T
    pos = 3.0 * (torch.Tensor(pos) / 32 - 0.5)

    for idx in range(10):
        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        grads = sf.get_score_z_given_z(
            sf.get_z_from_xu(pos, torch.zeros(pos.shape[0],2)), idx)[:,:2]
        grads = grads.detach().numpy()
        UV = np.swapaxes(grads, 0, 1).reshape(2, 32, 32)
        plt.quiver(X, Y, UV[0, :, :], UV[1, :, :])
        
        plt.subplot(1,2,2)
        grads = sf.get_score_z_given_z(
            sf.get_z_from_xu(torch.zeros(pos.shape[0],2), 0.2 * pos), idx)[:,2:]
        grads = grads.detach().numpy()
        UV = np.swapaxes(grads, 0, 1).reshape(2, 32, 32)
        plt.quiver(X, Y, UV[0, :, :], UV[1, :, :])
        
        plt.savefig("quiver_{:02d}.png".format(idx))
        plt.close()


if __name__ == "__main__":
    main()