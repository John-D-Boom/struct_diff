import torch
import numpy as np


def interpolate(x1:torch.Tensor, t):
    """
    Args:
        x1: Initial state
        t: Time step, [0,1]
    Returns a sample along the linear path between the noise and the initial state
    """
    noise = torch.randn(x1.shape)
    return t * x1 + (1-t) * noise

def reverse(x_hat, x_t, t):
    """
    Args:
        x_hat: The state at time t
        x_t: The state at time 0
        t: Time step, [0,1]
    Returns the state at time 1
    """
    