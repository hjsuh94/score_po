import numpy as np
import torch
import torch.optim as optim


class Policy:
    """
    Policy class. note that
    """

    def __init__(self, dim_x, dim_u):
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.dim_params = 0
        self.params = 0

    def get_parameters(self):
        return None

    def set_parameters(self, parameters_vector):
        return None

    def policy_jacobian(self, state, params_vector):
        return None

    def get_action(self, state, t, params_vector):
        return None

    def get_action_batch(self, state_batch, t, params_vector_batch):
        return None


class TimeVaryingOpenLoopPolicy(Policy):
    """
    Implement policy of the form
        u_t = pi_t.
    """

    def __init__(self, dim_x, dim_u, T):
        super().__init__(dim_x, dim_u)
        self.T = T
        self.dim_params = self.dim_u * self.T
        self.params_abstract = torch.zeros((T, self.dim_u))

    def get_parameters(self):
        return self.params_abstract.ravel()

    def set_parameters(self, params_vector):
        self.params_abstract = params_vector.reshape((self.T, self.dim_u))

    def policy_jacobian(self, state, t):
        jcb = torch.zeros((self.dim_u, self.T, self.dim_u))
        jcb[torch.arange(self.dim_u), t, torch.arange(self.dim_u)] = 1
        return jcb.reshape((self.dim_u, self.T * self.dim_u))

    def get_action(self, x, t):
        return self.params_abstract[t, :]

    def get_action_batch(self, x_batch, t):
        return self.params_abstract[t, :][None, :]


class TimeVaryingStateFeedbackPolicy(Policy):
    """
    Implement policy of the form
        u_t = K_t * x_t + mu_t.
    """

    def __init__(self, dim_x, dim_u, T):
        super().__init__(dim_x, dim_u)
        self.T = T
        self.dim_params = self.dim_u * (self.dim_x + 1) * self.T
        self.params = torch.zeros(self.T, self.dim_u, self.dim_x + 1)

    def get_parameters(self):
        return self.params.ravel()

    def set_parameters(self, params_vector):
        self.params = params_vector.reshape(self.T, self.dim_u, self.dim_x + 1)

    def policy_jacobian(self, state, t):
        """
        Get policy jacobian given state, parameter_vector and time t.
        Note that in index notation,
        u_i = K_ij x_j + m_i where K_ij is the gain and m_i is bias.
        Thus, du_i / d_K_jk = 0 if i != j and x_k if i = j
              du_i / d m_j = 0 if i != j and 1 if i = j
        """

        jcb = torch.zeros((self.dim_u, self.T, self.dim_u, self.dim_x + 1))
        jcb[torch.arange(self.dim_u), t, torch.arange(self.dim_u), : self.dim_x] = state
        jcb[torch.arange(self.dim_u), t, torch.arange(self.dim_u), self.dim_x :] = 1

        return jcb.reshape(self.dim_u, self.T * self.dim_u * (self.dim_x + 1))

    def policy_jacobian_batch(self, state_batch, t):
        """
        Get policy jacobian given state, parameter_vector and time t.
        Note that in index notation,
        u_i = K_ij x_j + m_i where K_ij is the gain and m_i is bias.
        Thus, du_i / d_K_jk = 0 if i != j and x_k if i = j
              du_i / d m_j = 0 if i != j and 1 if i = j
        """
        B = state_batch.shape[0]  # state_batch is of shape (B, n)
        # jacobian is of shape (B, u, T * u * (x + 1))
        jcb = torch.zeros((B, self.dim_u, self.T, self.dim_u, self.dim_x + 1))
        jcb[
            :, torch.arange(self.dim_u), t, torch.arange(self.dim_u), : self.dim_x
        ] = state_batch
        jcb[:, torch.arange(self.dim_u), t, torch.arange(self.dim_u), self.dim_x :] = 1

        return jcb.reshape(self.dim_u, self.T * self.dim_u * (self.dim_x + 1))

    def get_action(self, x, t):
        """
        Get action given current time and state.
        """
        gain = self.params[t, :, : self.dim_x]
        bias = self.params[t, :, self.dim_x :]

        return torch.einsum("ij,j->i", gain, x) + bias[:, 0]

    def get_action_batch(self, x_batch, t):
        """
        State_batch: B x dim_x
        gain
        """
        gain = self.params[t, :, : self.dim_x]
        bias = self.params[t, :, self.dim_x :]
        return np.einsum("ij,bj->bi", gain, x_batch) + bias[:, 0]


class NNPolicy(Policy):
    """
    Policy class. note that
    """

    def __init__(self, dim_x, dim_u, network):
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.net = network

        self.dim_params = len(self.net.get_vectorized_parameters())

    def get_parameters(self):
        return self.net.get_vectorized_parameters()

    def set_parameters(self, params_vector):
        return self.net.set_vectorized_parameters(params_vector)

    def policy_jacobian(self, x, t):
        """
        Particularly annoying and difficult to get for NN policy.
        """
        raise NotImplementedError("Policy Jacobian for NN is WIP.")

    def get_action(self, x, t):
        return self.net(x)

    def get_action_batch(self, x_batch, t):
        return self.net(x_batch)
