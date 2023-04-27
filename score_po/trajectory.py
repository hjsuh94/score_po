import abc
from enum import Enum
import os
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn


class Trajectory(torch.nn.Module):
    def __init__(self, dim_x, dim_u, T, x0):
        """
        Trajectory class for decision making.
        This class is for IVPs (Initial Value Problems),
        and has x0 has a fixed buffer, while 
        x_trj[1] to x_trj[T], u_trj[0] to u_trj[T-1] are decision variables.
        """
        super().__init__()
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.T = T
        self.register_buffer("x0", x0)
        self.declare_parameters()
        
    def declare_parameters(self):
        # x_trj: x_trj[1] to x_trj[T]
        # u_trj: u_trj[0] to u_trj[T-1]
        self.xnext_trj = nn.Parameter(torch.zeros((self.T, self.dim_x)))
        self.u_trj = nn.Parameter(torch.zeros((self.T, self.dim_u)))
        
    def get_full_trajectory(self):
        x_trj = torch.cat((self.x0, self.xnext_trj), dim=0)
        return x_trj, self.u_trj

    def forward(self, t):
        # The forward function returns state-input pairs at time t.
        # If t = 0, return x0, u0.
        if (t == 0):
            return self.x0, self.u_trj[t]
        # If t = T, return xT.
        elif (t == self.T):
            return self.xnext_trj[self.T-1]
        # Otherwise, return (xt, ut)
        else:
            return self.xnext_trj[t-1], self.u_trj[t]


class BVPTrajectory(Trajectory):
    def __init__(self, dim_x, dim_u, T, x0, xT):
        """
        Trajectory class for decision making.
        This class is for BVPs (Boundary Value Problems),
        and has x0, XT has a fixed buffer, while 
        x_trj[1] to x_trj[T-1], u_trj[0] to u_trj[T-1] are decision variables.
        """        
        super().__init__(dim_x, dim_u, T, x0)
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.T = T
        self.register_buffer("x0", x0)
        self.register_buffer("xT", xT)
        self.declare_parameters()        
        
    def declare_parameters(self):
        # x_trj: x_trj[1] to x_trj[T-1]
        # u_trj: u_trj[0] to u_trj[T-1]
        self.xnext_trj = nn.Parameter(torch.zeros((self.T-1, self.dim_x)))
        self.u_trj = nn.Parameter(torch.zeros((self.T, self.dim_u)))
        
    def get_full_trajectory(self):
        x_trj = torch.cat(
            (self.x0[None,:], self.xnext_trj, self.xT[None,:]), dim=0)
        return x_trj, self.u_trj

    def forward(self, t):
        # The forward function returns state-input pairs at time t.
        # If t = 0, return x0, u0.
        if (t == 0):
            return self.x0, self.u_trj[t]
        # If t = T, return xT.
        elif (t == self.T):
            return self.xT
        # Otherwise, return (xt, ut)
        else:
            return self.xnext_trj[t-1], self.u_trj[t]
