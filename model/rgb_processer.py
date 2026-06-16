import torch
import math
import torch.nn as nn   

from efficientnet_pytorch import EfficientNet
from typing import List, Dict, Optional, Tuple, Callable

class rgb_processer(nn.Module):
    def __init__(
        self,
        context_size: int = 5,  
        rgb_encoding_size: int = 256,
        mha_num_attention_heads: int = 2,
        mha_num_attention_layers: int = 2,
        mha_ff_dim_factor: int = 4,
        ) -> None:
        """
        Encode a sequence of RGB frames into a single 1D condition embedding.

        Args:
            context_size (int): number of RGB frames in the input sequence
            rgb_encoding_size (int): output embedding dimension
            mha_num_attention_heads (int): number of attention heads in the self-attention encoder
            mha_num_attention_layers (int): number of self-attention encoder layers
            mha_ff_dim_factor (int): feed-forward dimension multiplier relative to rgb_encoding_size
        """
        super().__init__()
        self.context_size = context_size 
        self.rgb_encoding_size = rgb_encoding_size     
        self.input_dim = 3 * 96 * 96
        self.rgb_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=3) # context
        self.rgb_encoder = replace_bn_with_gn(self.rgb_encoder)
        self.num_obs_features = self.rgb_encoder._fc.in_features
        self.rgb_encoder_type = "efficientnet"
    
        if self.num_obs_features != self.rgb_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.rgb_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()
              
        self.positional_encoding = PositionalEncoding(self.rgb_encoding_size, max_seq_len=self.context_size)
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=self.rgb_encoding_size, 
            nhead=mha_num_attention_heads, 
            dim_feedforward=mha_ff_dim_factor*self.rgb_encoding_size, 
            activation="gelu", 
            batch_first=True, 
            norm_first=True
        )
        self.sa_encoder = nn.TransformerEncoder(self.sa_layer, num_layers=mha_num_attention_layers)
    def forward(self, rgb: torch.tensor):
        """
        Args:
            rgb (torch.Tensor): RGB frames, shape (B, context_size*3, H, W)
        Returns:
            torch.Tensor: pooled condition embedding, shape (B, rgb_encoding_size)
        """
        # print("rgb shape: ", rgb.shape)
        rgb = torch.split(rgb, 3, dim=1)
        rgb = torch.cat(rgb, dim=0)  
        # print("rgb shape: ", rgb.shape)
        rgb_embedding = self.rgb_encoder.extract_features(rgb) 
        rgb_embedding = self.rgb_encoder._avg_pooling(rgb_embedding)
        if self.rgb_encoder._global_params.include_top:
            rgb_embedding = rgb_embedding.flatten(start_dim=1)
            rgb_embedding = self.rgb_encoder._dropout(rgb_embedding)
        rgb_embedding = self.compress_obs_enc(rgb_embedding)
        rgb_embedding = rgb_embedding.unsqueeze(1)
        rgb_embedding = rgb_embedding.reshape((self.context_size, -1, self.rgb_encoding_size))
        rgb_embedding = torch.transpose(rgb_embedding, 0, 1)    
        # print("rgb_embedding shape: ", rgb_embedding.shape)
        # Apply positional encoding 
        if self.positional_encoding:
            rgb_embedding = self.positional_encoding(rgb_embedding)

        # Apply self-attention
        rgb_embedding_fusion = self.sa_encoder(rgb_embedding)        
        rgb_embedding_fusion = torch.mean(rgb_embedding_fusion, dim=1)

        return rgb_embedding_fusion
    
    
       
def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Replace all BatchNorm layers in a module with GroupNorm.

    Args:
        root_module (nn.Module): module to modify in place
        features_per_group (int): number of channels per GroupNorm group
    Returns:
        nn.Module: the same module with BatchNorm2d layers replaced
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with the output of func.

    Args:
        root_module (nn.Module): module to modify in place
        predicate (Callable[[nn.Module], bool]): return True if the module should be replaced
        func (Callable[[nn.Module], nn.Module]): return the replacement module
    Returns:
        nn.Module: the same module with matching submodules replaced
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=6):
        """
        Precompute a sinusoidal positional encoding table.

        Args:
            d_model (int): embedding dimension
            max_seq_len (int): maximum sequence length to precompute encodings for
        """
        super().__init__()

        # Compute the positional encoding once
        pos_enc = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        pos_enc = pos_enc.unsqueeze(0)

        # Register the positional encoding as a buffer to avoid it being
        # considered a parameter when saving the model
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        """
        Add positional encoding to the input sequence.

        Args:
            x (torch.Tensor): input of shape (batch, seq_len, d_model)
        Returns:
            torch.Tensor: input with positional encoding added
        """
        x = x + self.pos_enc[:, :x.size(1), :]
        return x