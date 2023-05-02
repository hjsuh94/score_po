from omegaconf import DictConfig

import hydra
import torch

from score_po.nn import Normalizer
from score_po.score_matching import ScoreEstimatorXu
from examples.corridor.score_training import get_score_network, draw_score_result


@hydra.main(config_path="./config", config_name="optimize_path")
def main(cfg: DictConfig):
    network = get_score_network()
    score_estimator = ScoreEstimatorXu(
        dim_x=2,
        dim_u=2,
        network=network,
        z_normalizer=Normalizer(k=torch.ones(4), b=torch.zeros(4)),
    )
    score_estimator.load_state_dict(torch.load(cfg.score_estimator_load_ckpt))

    draw_score_result(
        score_estimator,
        cfg.device,
        cfg.corridor_width,
        cfg.horizontal_max,
        cfg.vertical_max,
        epsilon=1e-3,
        steps=10000,
    )


if __name__ == "__main__":
    main()
