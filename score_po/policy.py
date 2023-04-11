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

    def get_action(self, x, t):
        return self.params_abstract[t, :].to(x.device)

    def get_action_batch(self, x_batch, t):
        return self.params_abstract[t, :][None, :].to(x_batch.device)
    
    def save_parameters(self, filename):
        torch.save(self.params_abstract, filename)
        
    def load_parameters(self, filename):
        self.params_abstract = torch.load(filename)


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

    def get_action(self, x, t):
        """
        Get action given current time and state.
        """
        self.params = self.params.to(x.device)
        gain = self.params[t, :, : self.dim_x]
        bias = self.params[t, :, self.dim_x :]

        return torch.einsum("ij,j->i", gain, x) + bias[:, 0]

    def get_action_batch(self, x_batch, t):
        """
        State_batch: B x dim_x
        gain
        """
        self.params = self.params.to(x_batch.device)
        gain = self.params[t, :, : self.dim_x]
        bias = self.params[t, :, self.dim_x :]
        return torch.einsum("ij,bj->bi", gain, x_batch) + bias[:, 0]
    
    def save_parameters(self, filename):
        torch.save(self.params_abstract, filename)
        
    def load_parameters(self, filename):
        self.params_abstract = torch.load(filename)    


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
        self.net = self.net.to(params_vector.device)
        return self.net.set_vectorized_parameters(params_vector)

    def get_action(self, x, t):
        self.net = self.net.to(x.device)
        return self.net(x)

    def get_action_batch(self, x_batch, t):
        self.net = self.net.to(x_batch.device)
        return self.net(x_batch)
    
    def save_parameters(self, filename):
        self.net.save_network_parameters(filename)
        
    def load_parameters(self, filename):
        self.net.load_network_parameters(filename)
