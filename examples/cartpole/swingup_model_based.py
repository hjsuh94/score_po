"""
Use the analytical model to swing up the cartpole.
"""
from omegaconf import DictConfig, OmegaConf
import os
from typing import Optional

import hydra
import numpy as np
import matplotlib.pyplot as plt

from pydrake.solvers import MathematicalProgram, Solve
from examples.cartpole.cartpole_plant import CartpolePlant

OmegaConf.register_new_resolver("np.pi", lambda x: np.pi * x)


def traj_opt_model_based(
    nT: int,
    dt: float,
    x_lo: np.ndarray,
    x_up: np.ndarray,
    u_max: float,
    x_init: Optional[np.ndarray],
    u_init: Optional[np.ndarray],
):
    plant = CartpolePlant(dt)
    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(4, nT, "x")
    u = prog.NewContinuousVariables(1, nT - 1, "u")
    prog.AddBoundingBoxConstraint(
        np.zeros(4),
        np.zeros(4),
        x[:, 0],
    )
    prog.AddBoundingBoxConstraint(
        np.array([0, np.pi, 0, 0]), np.array([0, np.pi, 0, 0]), x[:, -1]
    )
    prog.AddBoundingBoxConstraint(-u_max, u_max, u)
    prog.AddBoundingBoxConstraint(
        x_lo.reshape((-1, 1)).repeat(nT, axis=1),
        x_up.reshape((-1, 1)).repeat(nT, axis=1),
        x,
    )

    for i in range(nT - 1):
        prog.AddConstraint(
            lambda xux: plant.dynamics(xux[:4], xux[4:5]) - xux[-4:],
            np.zeros(4),
            np.zeros(4),
            np.concatenate([x[:, i], u[:, i], x[:, i + 1]]),
        )
    prog.AddQuadraticCost(np.eye(nT - 1, nT - 1), np.zeros(nT - 1), u.squeeze())

    if x_init is not None:
        prog.SetInitialGuess(x, x_init)
    if u_init is not None:
        prog.SetInitialGuess(u, u_init)

    result = Solve(prog)
    print(result.get_solution_result())
    x_sol = result.GetSolution(x)
    u_sol = result.GetSolution(u)

    return x_sol, u_sol


@hydra.main(config_path="./config", config_name="learn_model")
def main(cfg: DictConfig):
    OmegaConf.save(cfg, os.path.join(os.getcwd(), "config.yaml"))
    nT = 60
    x_lo = np.array(cfg.nn_plant.x_lo)
    x_up = np.array(cfg.nn_plant.x_up)
    x_sol, u_sol = traj_opt_model_based(
        nT=nT,
        dt=cfg.nn_plant.dt,
        x_lo=x_lo,
        x_up=x_up,
        u_max=cfg.nn_plant.u_max,
        x_init=None,
        u_init=None,
    )
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(211)
    ax1.plot(x_sol[0, :], x_sol[1, :])
    ax1.set_xlabel("x")
    ax1.set_ylabel(r"$\theta$")
    ax2 = fig1.add_subplot(212)
    ax2.plot(x_sol[2, :], x_sol[3, :])
    ax2.set_xlabel(r"$\dot{x}$")
    ax2.set_ylabel(r"$\dot{\theta}$")
    fig1.savefig(os.path.join(os.getcwd(), "swingup_state.png"), format="png")

    fig2 = plt.figure()
    ax3 = fig2.add_subplot()
    ax3.plot(np.arange(nT - 1) * cfg.nn_plant.dt, u_sol.squeeze())
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel("u")
    fig2.savefig(os.path.join(os.getcwd(), "swingup_u.png"), format="png")


if __name__ == "__main__":
    main()
