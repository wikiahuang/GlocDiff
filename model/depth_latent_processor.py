from typing import Optional

import torch
import torch.nn as nn
import einops

from model.model_utils import PositionalEncoding
    
class depth_latent_processor(nn.Module):
    def __init__(
        self,
        context_size: int = 5,  
        depth_encoding_size: int = 256,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
        ) -> None:
        """
        Encode a sequence of 2D depth latents into a single 1D condition embedding.

        Args:
            context_size (int): number of depth latent frames in the input sequence
            depth_encoding_size (int): output embedding dimension
            mha_num_attention_heads (int): number of attention heads in the self-attention encoder
            mha_num_attention_layers (int): number of self-attention encoder layers
            mha_ff_dim_factor (int): feed-forward dimension multiplier relative to depth_encoding_size
        """
        super().__init__()
        self.context_size = context_size 
        self.depth_encoding_size = depth_encoding_size     
        self.input_dim = 4 * 96 * 96
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, depth_encoding_size)
        )
        self.positional_encoding = PositionalEncoding(self.depth_encoding_size, max_seq_len=self.context_size)
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=self.depth_encoding_size, 
            nhead=mha_num_attention_heads, 
            dim_feedforward=mha_ff_dim_factor*self.depth_encoding_size, 
            activation="gelu", 
            batch_first=True, 
            norm_first=True
        )
        self.sa_encoder = nn.TransformerEncoder(self.sa_layer, num_layers=mha_num_attention_layers)
    def forward(self, depth_latent: torch.tensor):
        """
        Args:
            depth_latent (torch.Tensor): depth latents, shape (B, context_size*4, 96, 96)
        Returns:
            torch.Tensor: pooled condition embedding, shape (B, depth_encoding_size)
        """
        # convert the depth latent to 1D latent
        b = depth_latent.shape[0]  
        depth_latent = einops.rearrange(depth_latent, 'b (l c) h w -> (b l) (c h w)', l = 5, c = 4)
        depth_latent = depth_latent.to(torch.float32)
        depth_embedding = self.mlp(depth_latent)
        depth_embedding = einops.rearrange(depth_embedding, '(b l) c -> b l c', b=b, l=5)

        # Apply positional encoding 
        if self.positional_encoding:
            depth_embedding = self.positional_encoding(depth_embedding)

        # Apply self-attention
        depth_embedding_fusion = self.sa_encoder(depth_embedding)        
        depth_embedding_fusion = torch.mean(depth_embedding_fusion, dim=1)

        return depth_embedding_fusion



