import torch.nn as nn

class glocdiff(nn.Module):
    def __init__(
        self,
        depth_latent_processor,
        # rgbprocesser,
        noise_pred_net,
        deep_floor_net
        ):
        """
        Wrap the depth latent processor, floor plan fusion network, and noise
        prediction network behind a single dispatch interface.

        Args:
            depth_latent_processor (nn.Module): encodes depth latents into a condition embedding
            noise_pred_net (nn.Module): predicts the diffusion noise residual
            deep_floor_net (nn.Module): fuses floor plan, pose, goal, and depth condition
        """
        super(glocdiff, self).__init__()
        self.depth_latent_processor = depth_latent_processor 
        # self.rgbprocesser = rgbprocesser
        self.noise_pred_net = noise_pred_net
        self.deep_floor_net = deep_floor_net    
       
    def forward(self, func_name, **kwargs):
        """
        Dispatch to one of the wrapped submodules by name.

        Args:
            func_name (str): which submodule to run ("noise_pred_net", "depth_latent_processor", or "deep_floor_net")
            **kwargs: keyword arguments forwarded to the selected submodule
        Returns:
            torch.Tensor: output of the selected submodule
        """
        if func_name == "noise_pred_net":
            output = self.noise_pred_net(sample=kwargs["sample"], timestep=kwargs["timestep"], global_cond=kwargs["global_cond"], local_cond=kwargs["local_cond"])
        elif func_name == "depth_latent_processor":
            output = self.depth_latent_processor(kwargs["depth_latent"])
        # elif func_name == "rgb_processer":
        #     output = self.rgbprocesser(kwargs["rgb"])
        elif func_name == "deep_floor_net":
            output = self.deep_floor_net(kwargs["floor_plan"], kwargs["cur_pos"], kwargs["cur_heading"], kwargs["goal_pos"], kwargs["depth_cond"])
        else:
            raise NotImplementedError
        return output