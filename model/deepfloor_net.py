import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable
from efficientnet_pytorch import EfficientNet
    
class deep_floor_net(nn.Module):
    def __init__(
        self,
        floorplan_encoder: Optional[str] = "efficientnet-b0",
        floorplan_encoding_size: Optional[int] = 256,
    ) -> None:
        """
        Encode a floor plan image and fuse it with pose, goal, and depth condition.

        Args:
            floorplan_encoder (str): backbone name used to encode the floor plan image
            floorplan_encoding_size (int): output embedding dimension
        """
        super().__init__()
        self.floorplan_encoding_size = floorplan_encoding_size
        self.floorplan_pos_ori_enc = nn.Linear(floorplan_encoding_size + 6, floorplan_encoding_size)
        # Initialize the observation encoder
        if floorplan_encoder.split("-")[0] == "efficientnet":
            self.floorplan_encoder = EfficientNet.from_name(floorplan_encoder, in_channels=3) # context
            self.floorplan_encoder = replace_bn_with_gn(self.floorplan_encoder)
            self.num_obs_features = self.floorplan_encoder._fc.in_features
            self.floorplan_encoder_type = "efficientnet"
        else:
            raise NotImplementedError
        
        
        #Initialize the depth_floorplan_fusion net
        self.fusion_net = nn.Linear(2*floorplan_encoding_size, floorplan_encoding_size)
        
        # self._initialize_weights()
        # Initialize the goal encoder
        # self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=6) # obs+goal
        # self.goal_encoder = replace_bn_with_gn(self.goal_encoder)
        # self.num_goal_features = self.goal_encoder._fc.in_features

        # Initialize compression layers if necessary
        if self.num_obs_features != self.floorplan_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.floorplan_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()
        
      
    def _initialize_weights(self):
        """
        Initialize fusion_net so it starts as an identity pass-through on the depth condition.
        """
        with torch.no_grad():
            
            self.fusion_net.weight[:, :self.floorplan_encoding_size] = torch.eye(self.floorplan_encoding_size)
            self.fusion_net.weight[:, self.floorplan_encoding_size:].fill_(0.0)
            
        
            nn.init.zeros_(self.fusion_net.bias)    

    def forward(self, floorplan: torch.tensor, obs_pos: torch.tensor, obs_ori: torch.tensor, goal_pos: torch.tensor, depth_latent: torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            floorplan (torch.Tensor): floor plan image, shape (B, 3, H, W)
            obs_pos (torch.Tensor): current position, shape (B, 2)
            obs_ori (torch.Tensor): current heading direction, shape (B, 2)
            goal_pos (torch.Tensor): goal position, shape (B, 2)
            depth_latent (torch.Tensor): depth condition embedding from depth_latent_processor, shape (B, floorplan_encoding_size)
        Returns:
            torch.Tensor: fused condition embedding, shape (B, floorplan_encoding_size)
        """
        floorplan_embedding = self.floorplan_encoder.extract_features(floorplan)
        floorplan_embedding = self.floorplan_encoder._avg_pooling(floorplan_embedding)
        if self.floorplan_encoder._global_params.include_top:
            floorplan_embedding = floorplan_embedding.flatten(start_dim=1)
            floorplan_embedding = self.floorplan_encoder._dropout(floorplan_embedding)
        floorplan_embedding = self.compress_obs_enc(floorplan_embedding)
        # print("floorplan_embedding", floorplan_embedding.shape)
        #add noise to obs_pos and obs_ori   
        # noisy_obs_pos = obs_pos + torch.normal(mean=0.1, std=0.02, size=obs_pos.shape).to(device)
        # noisy_obs_ori = obs_ori + torch.normal(mean=0.02, std=0.002, size=obs_ori.shape).to(device)   
        # print("floorplan_embedding", floorplan_embedding.shape)
        # print("noisy_obs_pos", noisy_obs_pos.shape)
        # print("noisy_obs_ori", noisy_obs_ori.shape)
        floorplan_pos_ori_embedding = self.floorplan_pos_ori_enc(torch.cat([floorplan_embedding, obs_pos, obs_ori, goal_pos], dim=1))
        deep_floorplan_fusion = self.fusion_net(torch.cat([depth_latent, floorplan_pos_ori_embedding], dim=1))
        
        return deep_floorplan_fusion
        
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



    