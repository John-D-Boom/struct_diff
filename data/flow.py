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

def reverse(x_t, x_hat, n, N):
    """
    Args:
        x_t: Current state
        x_hat: Model Prediction
        n: interval number
        N: Total number of steps
    Returns the state after reversing n steps
    """
    assert n < N
    assert n > 0

    t = n / N
    s = 1 / N + t

    return (s-t)/(1-t) * x_hat + (1-s)/(1-t) * x_t





    