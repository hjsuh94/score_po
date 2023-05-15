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
        self.warm_start = False
        self.initialize_params_for_mpc()

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
        if x_trj_last is not None:
            x_trj_now = torch.zeros_like(x_trj_last)
            x_trj_now[:-1] = x_trj_last[1:]
            x_trj_now[-1] = x_trj_last[-1]
            x_trj_now = x_trj_now.detach()
        else:
            x_trj_now = None

        u_trj_now = torch.zeros_like(u_trj_last)
        u_trj_now[:-1] = u_trj_last[1:]
        u_trj_now[-1] = u_trj_last[-1]
        u_trj_now = u_trj_now.detach()

        return x_trj_now, u_trj_now

    def reset_optimizer(self):
        self.opt.iter = 0

    def get_action(self, x):
        self.reset_optimizer()

        # reset initial condition.
        if isinstance(self.opt.trj, IVPTrajectory):
            self.opt.trj = IVPTrajectory(
                self.opt.trj.dim_x, self.opt.trj.dim_u, self.opt.trj.T, x.detach()
            )
        elif isinstance(self.opt.trj, SSTrajectory):
            self.opt.trj = SSTrajectory(
                self.opt.trj.dim_x, self.opt.trj.dim_u, self.opt.trj.T, x.detach()
            )
        else:
            raise NotImplementedError("MPC only supports SS and IVPTrajectory.")

        # warm start MPC from last iteration.
        if self.warm_start:
            if self.u_trj_last is None:
                x_trj_guess, u_trj_guess = None, None
            else:
                x_trj_guess, u_trj_guess = self.shift_trajectory(
                    self.x_trj_last, self.u_trj_last
                )
                k_x = self.opt.sf.x_normalizer.k
                k_u = self.opt.sf.u_normalizer.k
                if x_trj_guess is not None:
                    x_trj_guess += torch.randn_like(x_trj_guess) * 0.0 * k_x
                u_trj_guess += torch.randn_like(u_trj_guess) * 0.0
        else:
            x_trj_guess, u_trj_guess = None, None

        # iterate.
        self.opt.iterate(x_trj_guess=x_trj_guess, u_trj_guess=u_trj_guess)

        # get full trajectory.
        if isinstance(self.opt.trj, IVPTrajectory):
            self.x_trj_last, self.u_trj_last = self.opt.trj.get_full_trajectory()
        elif isinstance(self.opt.trj, SSTrajectory):
            self.u_trj_last = torch.clone(self.opt.trj.u_trj).detach()

        # return action.
        return self.opt.trj.u_trj[0]
