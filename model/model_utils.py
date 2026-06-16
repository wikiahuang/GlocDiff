import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=6):
        """
        Precompute a sinusoidal positional encoding table.

        Args:
            d_model (int): embedding dimension
            max_seq_len (int): maximum sequence length to precompute encodings for
        """
        super().__init__()
        pos_enc = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pos_enc', pos_enc.unsqueeze(0))

    def forward(self, x):
        """
        Add positional encoding to the input sequence.

        Args:
            x (torch.Tensor): input of shape (batch, seq_len, d_model)
        Returns:
            torch.Tensor: input with positional encoding added
        """
        return x + self.pos_enc[:, :x.size(1), :]
