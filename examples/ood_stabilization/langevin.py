import torch

def langevin(x0_batch, grad, step_size, iters):
    """
    grad is a function that takes in x_batch of shape (B, dim_x)
    and returns (B, dim_x).
    """
    B = x0_batch.shape[0]
    history = torch.zeros((iters, B, x0_batch.shape[1]))
    history[0] = x0_batch
    
    for iter in range(iters-1):
        history[iter + 1,:,:] = history[iter, :, :] - step_size * grad(
            history[iter, :, :]
        )
    return history
