import os, time
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb

from score_po.score_matching import ScoreEstimatorXux
from score_po.costs import Cost
from score_po.trajectory import Trajectory, BVPTrajectory
from score_po.nn import WandbParams, save_module, tensor_linspace


@dataclass
class TrajectoryOptimizerParams:
    cost: Cost
    sf: ScoreEstimatorXux
    trj: Trajectory
    T: int
    wandb_params: WandbParams    
    ivp: True # if false, we will assume bvp.
    lr: float = 1e-3
    beta: float = 0.0
    max_iters:int = 1000
    first_order: bool = True
    load_ckpt: Optional[str] = None
    save_best_model: Optional[str] = None
    saving_period: Optional[int] = 100
    device: str = "cuda"
    torch_optimizer: torch.optim.Optimizer = torch.optim.Adam

    def __init__(self):
        self.wandb_params = WandbParams()

    def load_from_config(self, cfg: DictConfig):
        self.T = cfg.trj.T
        self.lr = cfg.trj.lr
        self.max_iters = cfg.trj.max_iters
        self.save_best_model = cfg.trj.save_best_model
        self.saving_period = cfg.trj.saving_period
        self.load_ckpt = cfg.trj.load_ckpt
        self.device = cfg.trj.device
        self.beta = cfg.trj.beta
        self.wandb_params.load_from_config(cfg, field="trj")

    def to_device(self, device):
        self.cost.to(device)
        self.sf.to(self.device)
        self.trj.to(self.device)
        self.device = device


class TrajectoryOptimizerSF:
    def __init__(self, params: TrajectoryOptimizerParams, **kwargs):
        self.params = params
        self.cost = params.cost
        self.sf = params.sf
        self.trj = params.trj

    def get_value_loss(self):
        # Loop over trajectories to compute reward loss.
        x_trj, u_trj = self.trj.get_full_trajectory()
        cost = 0.0
        cost += self.cost.get_running_cost_batch(x_trj[:-1], u_trj[:]).sum()
        cost += self.cost.get_terminal_cost(x_trj[-1])
        return cost
    
    def modify_gradients(self, beta):
        # Modify value loss by applying score function.
        x_trj, u_trj = self.trj.get_full_trajectory()
        
        z_trj = torch.cat((x_trj[:-1], u_trj, x_trj[1:]), dim=1)
        sz_trj = self.sf.get_score_z_given_z(z_trj)
        sx_trj, su_trj, sxnext_trj = self.sf.get_xux_from_z(sz_trj)
        
        weight = -1 / self.params.sf.sigma ** 2
        
        if isinstance(self.trj, BVPTrajectory):
            self.trj.xnext_trj.grad += weight * beta * sx_trj[1:]
            self.trj.xnext_trj.grad += weight * beta * sxnext_trj[:-1]
            self.trj.u_trj.grad += weight * beta * su_trj
        else:
            self.trj.xnext_trj.grad[:-1] += weight * beta * sx_trj[1:]
            self.trj.xnext_trj.grad += weight * beta * sxnext_trj
            self.trj.u_trj.grad += weight * beta * su_trj            
            
    def initialize(self):
        xnext_trj_init = tensor_linspace(
            self.trj.x0, self.trj.xT, steps=self.trj.xnext_trj.shape[0]
        ).T
        xnext_trj_init += torch.randn_like(xnext_trj_init) * 0.1
        u_trj_init = torch.zeros(self.trj.u_trj.shape).to(self.params.device)
        u_trj_init += torch.randn_like(u_trj_init) * 0.1
        
        self.trj.xnext_trj = torch.nn.Parameter(xnext_trj_init)
        self.trj.u_trj = torch.nn.Parameter(u_trj_init)

    def iterate(self, callback=None):
        """
        Callback is a function that can be called with weightature 
        f(params, loss, iter)
        """
        if self.params.wandb_params.enabled:
            if self.params.wandb_params.dir is not None and not os.path.exists(
                self.params.wandb_params.dir
            ):
                self.os.makedirs(self.params.wandb_params.dir, exist_ok=True)
            wandb.init(
                project=self.params.wandb_params.project,
                name=self.params.wandb_params.name,
                dir=self.params.wandb_params.dir,
                config=self.params.wandb_params.config,
                entity=self.params.wandb_params.entity,
            )

        self.trj = self.trj.to(self.params.device)
        self.initialize()
        self.trj.train()

        loss = self.get_value_loss()

        start_time = time.time()
        print("Iteration: {:04d} | Cost: {:.3f} | Time: {:.3f}".format(0, loss, 0))
        self.cost_lst = np.zeros(self.params.max_iters)

        best_cost = np.inf
        self.cost_lst[0] = loss.item()

        optimizer = self.params.torch_optimizer(
            self.trj.parameters(), lr=self.params.lr
        )
        
        for iter in range(self.params.max_iters - 1):
            if callback is not None:
                callback(self.params, loss.item(), iter)
            
            optimizer.zero_grad()
            loss = self.get_value_loss()
            loss.backward()
            self.modify_gradients(self.params.beta)
            optimizer.step()
            self.cost_lst[iter + 1] = loss.item()

            if self.params.wandb_params.enabled:
                wandb.log({"trj_loss": loss.item()})
            if self.params.save_best_model is not None and (
                self.params.saving_period % self.params.saving_period == 0):
                model_path = os.path.join(os.getcwd(), self.params.save_best_model)
                model_dir = os.path.dirname(model_path)
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir, exist_ok=True)
                save_module(self.trj, model_path)

            print(
                "Iteration: {:04d} | Cost: {:.3f} | Time: {:.3f}".format(
                    iter + 1, loss.item(), time.time() - start_time
                )
            )

        return self.cost_lst
    
    def plot_iterations(self):
        cost_history_np = self.cost_history.clone().detach().numpy()
        plt.figure()
        plt.plot(np.arange(self.params.max_iters), cost_history_np)
        plt.xlabel("iterations")
        plt.ylabel("cost")
        plt.show()
        plt.close()
