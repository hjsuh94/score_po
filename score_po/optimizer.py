import time

import numpy as np
import torch
import matplotlib.pyplot as plt


class OptimizerParams:
    def __init__(self):
        self.lr = 1e-3
        self.iters = 1000
        self.batch_size = 512


class PolicyOptimizerParams:
    def __init__(self):
        self.cost = None  # Cost class.
        self.dynamical_system = None  # DynamicalSystem class.
        self.policy = None  # Policy class.

        self.policy_params_0 = None  # Initial guess for the policy.
        self.T = 20

        # These are used to sample across initial conditions.
        self.x0_upper = None
        self.x0_lower = None
        self.batch_size = 100

        # These are hyperparameters of policy search.
        self.std = 1e-1  # noise injected on output of policy.
        self.lr = 1e-3  # learning rate for gradient descent of policy search.
        self.max_iters = 200
        
        self.plot_iterations = True


class PolicyOptimizer:
    def __init__(self, params: PolicyOptimizerParams):
        self.params = params
        self.cost = params.cost
        self.ds = params.dynamical_system
        self.policy = params.policy

        self.policy_history = torch.zeros(
            (self.params.max_iters, self.policy.dim_params)
        )
        self.cost_history = torch.zeros(self.params.max_iters)
        # Set initial guess.
        self.policy_history[0, :] = self.params.policy_params_0
        self.policy.set_parameters(self.params.policy_params_0)

    def sample_initial_state_batch(self):
        """
        Given some batch size, sample initial states (B, dim_x).
        """
        initial = torch.rand(self.params.batch_size, self.ds.dim_x)
        return (
            self.params.x0_upper - self.params.x0_lower
        ) * initial + self.params.x0_lower

    def rollout_policy(self, x0, noise_trj):
        """
        Rollout policy, given
            x0: initial condition of shape (dim_x)
            noise_trj: (T, dim_u) noise output on the output of trajectory.
        """
        x_trj = torch.zeros((self.params.T + 1, self.ds.dim_x))
        u_trj = torch.zeros((self.params.T, self.ds.dim_x))
        x_trj[0] = x0

        for t in range(self.T):
            u_trj[t] = self.policy.get_action(x_trj[t], t) + noise_trj[t]
            x_trj[t + 1] = self.ds.dynamics(x_trj[t], u_trj[t])

        return x_trj, u_trj

    def rollout_policy_batch(self, x0_batch, noise_trj_batch):
        """
        Rollout policy in batch, given
            x0_batch: initial condition of shape (B, dim_x)
            noise_trj_batch: (B, T, dim_u) noise output on the output of trajectory.
        """
        B = x0_batch.shape[0]
        x_trj_batch = torch.zeros((B, 0, self.ds.dim_x))
        u_trj_batch = torch.zeros((B, 0, self.ds.dim_u))
        x_trj_batch = torch.hstack((x_trj_batch, x0_batch[:,None,:]))

        for t in range(self.params.T):
            u_t_batch = self.policy.get_action_batch(
                x_trj_batch[:, t, :], t) + noise_trj_batch[:, t, :]
            u_trj_batch = torch.hstack((u_trj_batch, u_t_batch[:,None,:]))
            x_next_batch = self.ds.dynamics_batch(
                x_trj_batch[:, t, :], u_trj_batch[:, t, :])
            x_trj_batch = torch.hstack((x_trj_batch, x_next_batch[:,None,:]))

        return x_trj_batch, u_trj_batch

    def evaluate_cost(self, x0, noise_trj):
        """
        Evaluate the cost given:
            x0: initial condition of shape (dim_x)
            noise_trj: (T, dim_u) noise output on the output of trajectory.
        """
        x_trj, u_trj = self.rollout_policy(x0, noise_trj)
        cost = 0.0
        for t in range(self.T):
            cost += self.cost.get_running_cost(x_trj[t], u_trj[t])
        cost += self.cost.get_terminal_cost(x_trj[self.params.T])
        return cost

    def evaluate_cost_batch(self, x0_batch, noise_trj_batch):
        """
        Evaluate the cost given:
            x0: initial condition of shape (dim_x)
            noise_trj: (T, dim_u) noise output on the output of trajectory.
        """
        x_trj_batch, u_trj_batch = self.rollout_policy_batch(x0_batch, noise_trj_batch)
        B = x_trj_batch.shape[0]
        cost = torch.zeros(B)
        for t in range(self.params.T):
            cost += self.cost.get_running_cost_batch(
                x_trj_batch[:, t, :], u_trj_batch[:, t, :]
            )
        cost += self.cost.get_terminal_cost_batch(x_trj_batch[:, self.params.T, :])
        return cost

    def get_policy_gradient(self, x0_batch):
        """
        Obtain the policy gradient of the value function starting from state x0.
        """
        raise NotImplementedError("This function is virtual.")

    def iterate(self):
        zero_noise_trj = torch.zeros(
            (self.params.batch_size, self.params.T, self.ds.dim_u)
        )
        x0_batch = self.sample_initial_state_batch()
        cost = torch.mean(self.evaluate_cost_batch(x0_batch, zero_noise_trj))
        self.cost_history[0] = cost

        start_time = time.time()
        print("Iteration: {:04d} | Cost: {:.3f} | Time: {:.3f}".format(0, cost, 0))

        for iter in range(self.params.max_iters - 1):
            x0_batch = self.sample_initial_state_batch()

            self.policy_history[iter + 1] = self.policy_history[
                iter
            ] - self.params.lr * self.get_policy_gradient(x0_batch)

            self.policy.set_parameters(self.policy_history[iter + 1])

            cost = torch.mean(self.evaluate_cost_batch(x0_batch, zero_noise_trj))
            self.cost_history[iter + 1] = cost

            print(
                "Iteration: {:04d} | Cost: {:.3f} | Time: {:.3f}".format(
                    iter + 1, cost, time.time() - start_time
                )
            )

        if self.params.plot_iterations:
            cost_history_np = self.cost_history.clone().detach().numpy()
            plt.figure()
            plt.plot(np.arange(self.params.max_iters), cost_history_np)
            plt.xlabel("iterations")
            plt.ylabel("cost")
            plt.savefig("test.png")
            plt.close()

        return self.policy_history


class FirstOrderPolicyOptimizer(PolicyOptimizer):
    def __init__(self, params: PolicyOptimizerParams):
        super().__init__(params)

    def get_policy_gradient(self, x0_batch):
        """
        Given x0_batch, obtain the policy gradients.
        """
        B = x0_batch.shape[0]
        noise_trj_batch = torch.normal(
            0, self.params.std, size=(B, self.params.T, self.ds.dim_u)
        )

        # Initiate autodiff.
        params = self.policy_history[iter]
        params.requires_grad = True
        self.policy.set_parameters(params)

        cost_mean = torch.mean(self.evaluate_cost_batch(x0_batch, noise_trj_batch))
        cost_mean.backward()

        return params.grad


class FirstOrderNNPolicyOptimizer(PolicyOptimizer):
    """
    First order optimizer when the policy is NN.
    We need special treatment because of how we can't pass on autodiff to parameters
    of the neural nets.
    """

    def __init__(self, params: PolicyOptimizerParams):
        super().__init__(params)

    def get_policy_gradient(self, x0_batch):
        """
        Given x0_batch, obtain the policy gradients.
        """
        B = x0_batch.shape[0]
        noise_trj_batch = torch.normal(
            0, self.params.std, size=(B, self.params.T, self.ds.dim_u)
        )

        # Initiate autodiff.
        self.policy.net.train()
        self.policy.net.zero_grad()

        torch.autograd.set_detect_anomaly(True)

        cost_mean = torch.mean(self.evaluate_cost_batch(x0_batch, noise_trj_batch))
        cost_mean.backward()

        return self.policy.net.get_vectorized_gradients()
