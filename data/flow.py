import torch
import numpy as np


def interpolate(x1:torch.Tensor, t, mask = None, device = None, source_scale = 1.0):
    """
    Args:
        x1: Initial state
        t: Time step, [0,1]
        mask: tensor with same shape as x1, 0 where padding, 1 where data
        source_scale: float to scale the gaussian by.
    Returns a sample along the linear path between the noise and the initial state
    """
    assert 0 <= t <= 1

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise = source_scale * torch.randn(x1.shape, device = device)

    if mask is not None:
        #Deals with batching, because don't want to create gradient on the padding
        noise = noise * mask
    x_t = t * x1 + (1-t) * noise
    return x_t, noise

def reverse_euler(x_t, v_model, dt, ):
    """
    Args:
        x_t: Current state
        v_model: Model prediction of conditional flow
        dt: Time step
    """
    return x_t + v_model * dt





    
