import os, time
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb

from score_po.dynamical_system import DynamicalSystem
from score_po.score_matching import (
    ScoreEstimatorXux,
    ScoreEstimatorXu,
    NoiseConditionedScoreEstimatorXu,
    NoiseConditionedScoreEstimatorXux,
)
from score_po.data_distance import DataDistanceEstimatorXux
from score_po.costs import Cost
from score_po.trajectory import Trajectory, IVPTrajectory, BVPTrajectory, SSTrajectory
from score_po.nn import WandbParams, save_module, tensor_linspace
from score_po.trajectory_optimizer import TrajectoryOptimizer


class MPC:
    def __init__(self, opt: TrajectoryOptimizer):
        self.opt = opt
        self.x_trj_last = None
        self.u_trj_last = None
        self.iter = 0
        self.initialize_params_for_mpc()
        if isinstance(opt.trj, BVPTrajectory):
            raise NotImplementedError(
                "Due to handling recursive feasibility, we assume each MPC is solving a IVP problem."
            )

    def initialize_params_for_mpc(self):
        # Avoid every loggin operations.
        self.opt.params.wandb_params.enabled = False
        self.opt.params.saving_period = None
        self.opt.params.save_best_model = None
        self.opt.params.verbose = False

    def shift_trajectory(self, x_trj_last, u_trj_last):
        """
        From x0 ~ xT, compute a guess of x1 ~ xT+1 by shifting the trajectories by one.
        """
        x_trj_now = torch.zeros_like(x_trj_last)
        x_trj_now[:-1] = x_trj_last[1:]
        x_trj_now[-1] = x_trj_last[-1]

        u_trj_now = torch.zeros_like(u_trj_last)
        u_trj_now[:-1] = u_trj_last[1:]
        u_trj_now[-1] = u_trj_last[-1]

        return x_trj_now.detach(), u_trj_now.detach()

    def reset_optimizer(self):
        self.opt.iter = 0

    def get_action(self, x):
        self.reset_optimizer()

        # reset initial condition.
        self.opt.trj = IVPTrajectory(2, 2, self.opt.trj.T, x.detach())
        # warm start MPC from last iteration.
        if self.x_trj_last is None:
            x_trj_guess, u_trj_guess = None, None
        else:
            x_trj_guess, u_trj_guess = self.shift_trajectory(
                self.x_trj_last.detach(), self.u_trj_last.detach()
            )

        # iterate.
        self.opt.iterate(x_trj_guess=x_trj_guess, u_trj_guess=u_trj_guess)

        # get full trajectory.
        self.x_trj_last, self.u_trj_last = self.opt.trj.get_full_trajectory()

        # return action.
        return self.opt.trj.u_trj[0]
