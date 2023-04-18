import torch
import numpy as np
import pickle
from torch.utils.data import TensorDataset

import hydra
from omegaconf import DictConfig

from score_po.dynamical_system import NNDynamicalSystem
from score_po.nn import MLP, train_network, TrainParams
from box_pushing_system import PlanarPusherSystem


@hydra.main(config_path="./config", config_name="dynamics_training")
def main(cfg: DictConfig):
    with open(cfg.dataset_dir, "rb") as f:
        dataset = pickle.load(f)

    x = torch.Tensor(dataset["x"])
    xnext = torch.Tensor(dataset["xnext"])
    keypts_x = torch.Tensor(dataset["keypts_x"])
    keypts_xnext = torch.Tensor(dataset["keypts_xnext"])
    u = torch.Tensor(dataset["u"])

    # We need to append pusher coordinates to keypoints to fully define
    # the state.
    new_x = torch.hstack((keypts_x, x[:, 3:5]))
    print(new_x.shape)
    new_xnext = torch.hstack((keypts_xnext, xnext[:, 3:5]))

    dataset = TensorDataset(new_x, u, new_xnext)
    network = MLP(14, 12, cfg.nn_layers)
    params = TrainParams()
    params.load_from_config(cfg)

    nn_dynamics = NNDynamicalSystem(dim_x=12, dim_u=2, network=network)
    nn_dynamics.train_network(dataset, params)


if __name__ == "__main__":
    main()
