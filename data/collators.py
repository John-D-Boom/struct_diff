import torch
import numpy as np
import os
from torch.nn.utils.rnn import pad_sequence

def custom_pad(batch_tuple, pad_value = 0):
    
    
    padded = pad_sequence(torch.tensor(tens_list), batch_first=True, padding_value= pad_value)
    
    masks = padded == pad_value
    return (padded, masks)