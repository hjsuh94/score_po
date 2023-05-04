import torch
import numpy as np
from tqdm import tqdm
import mujoco_py
import cv2
import gym, d4rl
import os
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from score_po.trajectory import Trajectory, IVPTrajectory, BVPTrajectory, SSTrajectory
from score_po.policy import Policy, NNPolicy
from score_po.nn import MLP, TrainParams, Normalizer
from score_po.policy_optimizer import (
    DRiskScorePolicyOptimizer,
    DRiskScorePolicyOptimizerParams,
)
from score_po.mpc import MPC


class TrajectoryVisualizer:
    """Debugging tool for visualizing trajectory."""

    def __init__(self, env_name: str):
        self.env = gym.make(env_name)
        self.env.reset()

    def set_obs(self, obs):
        obs = np.concatenate((np.zeros(1), obs))
        q_pos = obs[: int(len(obs) / 2)]
        q_vel = obs[int(len(obs) / 2) :]
        self.env.set_state(q_pos, q_vel)

    def render_image(self, obs, filename):
        self.set_obs(obs)
        img = self.env.sim.render(width=900, height=900)
        img = cv2.flip(img, 0)
        img = cv2.cvtColor(img, 4)
        img = img[300:600, 300:600]
        cv2.imwrite(filename, img)

    def render_trajectory(self, x_trj, u_trj, foldername):
        if not os.path.isdir(foldername):
            os.mkdir(foldername)

        hal_dir = os.path.join(foldername, "hallucinated")
        if not os.path.isdir(hal_dir):
            os.mkdir(hal_dir)

        real_dir = os.path.join(foldername, "real")
        if not os.path.isdir(real_dir):
            os.mkdir(real_dir)

        for t in range(u_trj.shape[0]):
            self.render_image(
                x_trj[t],
                os.path.join(hal_dir, "x_trj_{:05d}_hallucinated.png".format(t)),
            )

            # Get real trajectory.
            self.set_obs(x_trj[t])
            x_true, _, _, _ = self.env.step(u_trj[t])
            self.render_image(
                x_true, os.path.join(real_dir, "x_trj_{:05d}_real.png".format(t + 1))
            )


class GymPolicyEvaluator:
    """Given environment and policy, evaluate the performance."""

    def __init__(self, env_name: str, mpc: MPC):
        self.env = gym.make(env_name)
        self.dim_x = self.env.observation_space.shape[0]
        self.dim_u = self.env.action_space.shape[0]
        self.mpc = mpc
        # Taken from D4RL paper, page 7, evaluation protocol.
        self.batch_size = 1
        # self.video_recorder = VideoRecorder(self.env, "output.mp4", enabled=True)
        self.traj_vis = TrajectoryVisualizer(env_name)

    def check_input_consistency(self):
        assert self.policy.dim_x == self.dim_x
        assert self.policy.dim_u == self.dim_u

    def render_image(self, filename):
        img = self.env.sim.render(width=900, height=900)
        img = cv2.flip(img, 0)
        img = cv2.cvtColor(img, 4)
        img = img[300:600, 300:600]
        cv2.imwrite(filename, img)

    def get_policy_score(self):
        obs = self.env.reset()
        done = False
        time = 0
        returns = 0.0

        if not os.path.isdir("true"):
            os.mkdir("true")

        while not done:
            self.render_image("true/{:05d}.png".format(time))
            action = self.mpc.get_action(
                torch.Tensor(obs).to(self.mpc.opt.params.device)
            )

            if isinstance(self.mpc.opt.trj, BVPTrajectory):
                x_trj_last = self.mpc.x_trj_last.cpu().detach().numpy()
                u_trj_last = self.mpc.u_trj_last.cpu().detach().numpy()
            else:
                x_trj_last, u_trj_last = self.mpc.opt.rollout_trajectory()
                x_trj_last = x_trj_last.cpu().detach().numpy()
                u_trj_last = u_trj_last.cpu().detach().numpy()

            self.traj_vis.render_trajectory(
                x_trj_last, u_trj_last, "{:04d}".format(time)
            )
            obs, reward, done, _ = self.env.step(action.cpu().detach().numpy())

            time += 1
            returns += reward
            print(returns)
            # print(self.env.get_normalized_score(returns))
        return self.env.get_normalized_score(returns)

    def get_policy_score_mean(self):
        policy_scores = np.zeros(self.batch_size)
        for b in tqdm(range(self.batch_size)):
            policy_scores[b] = self.get_policy_score()
        return np.mean(policy_scores)
