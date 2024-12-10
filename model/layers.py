import torch
import numpy as np
import torch.nn as nn

class TransformerLayer(nn.Module):
    def __init__(self, d_model=1280, num_heads=8, hidden_size = 1, dropout=0.1):
        """
        Transformer layer that maintains input dimensions [B, N, D]
        
        Args:
        - d_model (int): Dimension of the model (input/output dimension)
        - num_heads (int): Number of attention heads
        - dropout (float): Dropout probability
        """
        super().__init__()

        assert isinstance(hidden_size, int), "size must be of type int"
        
        # Multi-Head Self-Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * hidden_size, d_model)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass of the transformer layer
        
        Args:
        - x (Tensor): Input tensor of shape [B, N, D]
        
        Returns:
        - Tensor: Output tensor of shape [B, N, D]
        """
        # Self-Attention
        attn_output, _ = self.self_attn(x, x, x)
        
        # Residual connection and layer normalization
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward network
        ffn_output = self.ffn(x)
        
        # Final residual connection and layer normalization
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x

def time_encoding(time, seq_len = 300, d_model = 1280):
    """
    Generate a sinusoidal embedding for a continuous time value between 0 and 1.
    
    Args:
    - time (float): Continuous time value between 0 and 1
    - d_model (int): Dimension of the model's embeddings
    - seq_len (int): Length of the input sequence to match
    Returns:
    - torch.Tensor: Embedding vector of shape (seq_len, d_model), where each row is the same
    """

    max_time_increment = 1000 #Defines precision of timestep. This means timestep rounded to 1/1000th

    # Validate input
    assert 0 <= time <= 1, "Time must be between 0 and 1"
    # Convert time to discrete position
    pos = int(time * (max_time_increment - 1))
    
    # Generate positional encoding
    position = torch.arange(max_time_increment, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(7000.0) / d_model))
    
    pe = torch.zeros(max_time_increment, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe[pos].expand(seq_len, -1)


def positional_encoding(max_seq_len, d_model):
    """
    Generate sinusoidal positional encoding for a sequence of a given length and dimension.
    Matches the implementation in the "Attention is All You Need" paper from Vaswani et al, 2017.
    
    Args:
    - max_seq_len (int): Maximum sequence length
    - d_model (int): Dimension of the model's embeddings
    
    Returns:
    - torch.Tensor: Positional encoding matrix of shape (max_seq_len, d_model)
    """
    position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (- np.log(10000.0) / d_model))
    
    pe = torch.zeros(max_seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe

