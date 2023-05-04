from omegaconf import DictConfig, OmegaConf
import os

import hydra
import torch

from examples.cartpole.cartpole_plant import CartpolePlant, CartpoleNNDynamicalSystem
from examples.cartpole.score_training import get_score_network, draw_score_result
from score_po.score_matching import (
    ScoreEstimatorXu,
    ScoreEstimatorXux,
    NoiseConditionedScoreEstimatorXux,
)


@hydra.main(config_path="./config", config_name="swingup")
def main(cfg: DictConfig):
    OmegaConf.save(cfg, os.path.join(os.getcwd(), "config.yaml"))
    device = cfg.device

    nn_plant = CartpoleNNDynamicalSystem(
        hidden_layers=cfg.plant_param.hidden_layers, device=device
    )
    nn_plant.load_state_dict(torch.load(cfg.dynamics_load_ckpt))
    nn_plant.to(cfg.device)

    score_network = get_score_network(cfg.single_shooting)
    score_estimator_cls = (
        ScoreEstimatorXu if cfg.single_shooting else NoiseConditionedScoreEstimatorXux
    )
    score_estimator = score_estimator_cls(dim_x=4, dim_u=1, network=score_network).to(
        cfg.device
    )
    if cfg.single_shooting:
        load_ckpt = cfg.score_estimator_xu_load_ckpt
    else:
        load_ckpt = cfg.score_estimator_xux_load_ckpt
    score_estimator.load_state_dict(torch.load(load_ckpt))

    plant = CartpolePlant(dt=cfg.plant_param.dt)

    for eps in (1e-1, 1e-2, 1e-3):
        for steps in (30000,):
            draw_score_result(
                score_estimator,
                plant,
                nn_plant,
                cfg.device,
                epsilon=eps,
                steps=steps,
                x_lb=torch.tensor(cfg.plant_param.x_lo),
                x_ub=torch.tensor(cfg.plant_param.x_up),
            )


if __name__ == "__main__":
    main()
