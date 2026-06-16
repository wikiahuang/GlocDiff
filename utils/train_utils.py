
import os
import time
import numpy as np
import lmdb
import tqdm

from Logger import Logger

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.distributed as dist
import matplotlib.pyplot as plt


ACTION_STATS = {}
ACTION_STATS["min"] = np.array([-2.5, -4])
ACTION_STATS["max"] = np.array([5, 4])    
RED = np.array([1, 0, 0])
GREEN = np.array([0, 1, 0])
CYAN = np.array([0, 1, 1])
MAGENTA = np.array([1, 0, 1])


# Train utils for NOMAD

def _compute_losses(
    model,
    noise_scheduler,
    depth_cond,
    shortest_actions,
    collision_free_actions,
    # metric_goal,
    device: torch.device,
):
    """
    Run a full diffusion denoising pass and compare it against the ground-truth actions.

    Args:
        model (nn.Module): glocdiff model
        noise_scheduler (DDPMScheduler): diffusion noise scheduler
        depth_cond (torch.Tensor): fused condition embedding, shape (B, encoding_size)
        shortest_actions (torch.Tensor): local-condition actions, shape (B, len_traj_pred, 2)
        collision_free_actions (torch.Tensor): ground-truth actions, shape (B, len_traj_pred, 2)
        device (torch.device): device to run inference on
    Returns:
        dict: action_loss, action_waypts_cos_sim, and multi_action_waypts_cos_sim
    """


    pred_horizon, action_dim = collision_free_actions.shape[1], collision_free_actions.shape[2] 
    model_output_dict = model_output(
        model,
        noise_scheduler,
        depth_cond,
        shortest_actions,   
        pred_horizon,
        action_dim,
        # metric_goal,
        device=device,
    )
    actions = model_output_dict['actions']
    rank = int(os.environ['LOCAL_RANK'])
    collision_free_actions = collision_free_actions.to(rank)
    def action_reduce(unreduced_loss: torch.Tensor):
        """
        Reduce a per-element loss tensor down to a single scalar by averaging.

        Args:
            unreduced_loss (torch.Tensor): per-batch, per-element loss values
        Returns:
            torch.Tensor: scalar mean loss
        """
        # Reduce over non-batch dimensions to get loss per batch element
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        return unreduced_loss.mean()


    action_loss = action_reduce(F.mse_loss(actions, collision_free_actions, reduction="none"))

    action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
        actions[:, :, :2], collision_free_actions[:, :, :2], dim=-1
    ))

    multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
        torch.flatten(actions[:, :, :2], start_dim=1),
        torch.flatten(collision_free_actions[:, :, :2], start_dim=1),
        dim=-1,
    ))

    results = {
        "action_loss": action_loss,
        "action_waypts_cos_sim": action_waypts_cos_similairity,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim,
    }

    return results

def visualize_trajectory(
    model: nn.Module,
    noise_scheduler: DDPMScheduler,
    depth_cond: torch.Tensor,
    local_cond: torch.Tensor,
    gt_actions: torch.Tensor,
    num_images: int = 4,
    rank: int = 0,
) -> plt.Figure:
    """
    Run full diffusion denoising on a few samples and plot predicted vs. ground-truth trajectories.

    Args:
        model (nn.Module): glocdiff model
        noise_scheduler (DDPMScheduler): diffusion noise scheduler
        depth_cond (torch.Tensor): fused condition embedding, shape (B, encoding_size)
        local_cond (torch.Tensor): local-condition actions, shape (B, len_traj_pred, 2)
        gt_actions (torch.Tensor): ground-truth waypoints, shape (B, len_traj_pred, 2)
        num_images (int): number of samples to visualize
        rank (int): local GPU rank to run inference on
    Returns:
        plt.Figure: figure with one subplot per visualized sample
    """
    num_vis = min(num_images, depth_cond.shape[0])
    len_traj = gt_actions.shape[1]

    model.eval()
    with torch.no_grad():
        noisy_output = torch.randn((num_vis, len_traj, 2), device=f'cuda:{rank}')
        noise_scheduler.set_timesteps(noise_scheduler.config.num_train_timesteps)
        for t in noise_scheduler.timesteps:
            noise_pred = model(
                "noise_pred_net",
                sample=noisy_output,
                timestep=t.unsqueeze(0).expand(num_vis).to(rank),
                global_cond=depth_cond[:num_vis],
                local_cond=local_cond[:num_vis],
            )
            noisy_output = noise_scheduler.step(
                model_output=noise_pred, timestep=t, sample=noisy_output
            ).prev_sample
    model.train()

    ndeltas = noisy_output.cpu().numpy()
    ndeltas = (ndeltas + 1) / 2
    ndeltas = ndeltas * (ACTION_STATS['max'] - ACTION_STATS['min']) + ACTION_STATS['min']
    pred_waypoints = np.cumsum(ndeltas, axis=1)
    gt_waypoints = gt_actions[:num_vis].cpu().numpy()

    fig, axes = plt.subplots(1, num_vis, figsize=(4 * num_vis, 4))
    if num_vis == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(gt_waypoints[i, :, 0], gt_waypoints[i, :, 1], 'b-o', markersize=3, label='GT')
        ax.plot(pred_waypoints[i, :, 0], pred_waypoints[i, :, 1], 'r-o', markersize=3, label='Pred')
        ax.set_aspect('equal')
        ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def action_reduce(unreduced_loss: torch.Tensor) -> torch.Tensor:
    """
    Reduce a per-element loss tensor down to a single scalar by averaging.

    Args:
        unreduced_loss (torch.Tensor): per-batch, per-element loss values
    Returns:
        torch.Tensor: scalar mean loss
    """
    while unreduced_loss.dim() > 1:
        unreduced_loss = unreduced_loss.mean(dim=-1)
    return unreduced_loss.mean()


def train_one_epoch(
    model: nn.Module,
    optimizer: Adam,
    dataloader: DataLoader,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
    depth_generator,
    epoch: int,
    data_folder: str,
    alpha: float = 1e-4,
    beta: float = 1,
    print_log_freq: int = 100,
    image_log_freq: int = 50,
    num_images_log: int = 8,
    logger: str = "none",
    epochs: int = 1,
    max_sigma: float = 0.01,
    writter=None,   
    metric_waypoint_spacing: float = 0.045,
    waypoint_spacing: float = 1,
    # dictpath: str=None  
):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): model to train
        optimizer (Adam): optimizer to use
        dataloader (DataLoader): dataloader for training
        device (torch.device): device to use
        noise_scheduler (DDPMScheduler): diffusion noise scheduler
        depth_generator: pretrained Marigold depth pipeline used on cache misses
        epoch (int): current epoch
        data_folder (str): root folder containing the scene data and depth latent cache
        alpha (float): weight of the action loss
        beta (float): weight of the distance loss
        print_log_freq (int): how often (in batches) to log scalar metrics
        image_log_freq (int): how often (in batches) to log trajectory visualizations
        num_images_log (int): number of samples to visualize per trajectory plot
        logger (str): which logging backend to use ("tensorboard", "wandb", or "none")
        epochs (int): total number of epochs (used by the noise curriculum)
        max_sigma (float): maximum noise amplitude for the local-condition curriculum noise
        writter: TensorBoard SummaryWriter, or None if not using TensorBoard
        metric_waypoint_spacing (float): meters per waypoint unit
        waypoint_spacing (int): number of frames between consecutive waypoints
    """
    model.train()
    # Freeze depth generator
    num_batches = len(dataloader)
    
    
    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)
    action_waypts_cos_sim_logger = Logger(
        "action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    multi_action_waypts_cos_sim_logger = Logger(
        "multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    loggers = {
        "action_loss": action_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim_logger,
    }
    rank = int(os.environ['LOCAL_RANK'])
    train_data_path = os.path.join(data_folder, "train")
    lmdb_file = os.path.join(train_data_path, "depth0.lmdb")
    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for batch_index, data in enumerate(tepoch):
            (
                strs,
                obs_image, 
                shortest_actions,
                collision_free_actions,
                meter_collision_free_actions,  
                floor_plan,
                metric_goal,
                traversable_map_path
            ) = data         
            # print("shape of metric goal:", metric_goal.size())
            metric_goal = metric_goal.to(rank)
            cur_pos = meter_collision_free_actions[:, 0, :2]
            cur_heading = meter_collision_free_actions[:, 0, 2:]
            start_time = time.time()    
            batch_size, c = obs_image.shape[0], obs_image.shape[1]
            context_size = c // 3
            #1. get 2D depth latent using pretrained marigold model
            depth_latent_tuple = ()
            for sample_index in range(batch_size):              
                scene, traj, current_time = strs[sample_index].split('-')
                context_times =  list(
                    range(
                        int(current_time) - context_size * waypoint_spacing + 1,
                        int(current_time) + 1,
                        waypoint_spacing,
                    )
                )      
                with lmdb.open(lmdb_file, map_size=2**40, lock=True, readahead=True, meminit=False) as latent_cache:
                    lmdb_keys = [
                        os.path.join(train_data_path, scene, traj, str(t).zfill(5)).encode()
                        for t in context_times
                    ]
                    # Read-only pass: cache hits don't need to wait on other processes' writes
                    with latent_cache.begin(write=False) as txn:
                        cached_values = [txn.get(k) for k in lmdb_keys]

                    latent_tuple = ()
                    pending_writes = {}
                    for i, context_time in enumerate(context_times):
                        cached = cached_values[i]
                        if cached is not None:
                            depth_latent = np.frombuffer(cached, dtype=np.float16)
                            depth_latent = depth_latent.reshape(1, 4, 96, 96)
                            depth_latent = from_numpy(depth_latent).to(rank)
                        else:
                            obs = torch.split(obs_image[sample_index], 3, dim=0)[i]
                            output_list = depth_generator(obs, output_latent=True)
                            depth_latent = output_list.latent
                            pending_writes[lmdb_keys[i]] = depth_latent.cpu().numpy().tobytes()
                        latent_tuple += (depth_latent,)

                    # Write-only pass: only taken when there's something new to cache
                    if pending_writes:
                        with latent_cache.begin(write=True) as txn:
                            for k, v in pending_writes.items():
                                txn.put(k, v)
                    depth_latent_i = torch.cat(latent_tuple, dim=1) #[L * c, h, w]
                depth_latent_tuple += (depth_latent_i,)
            depth_latent = torch.cat(depth_latent_tuple, dim=0) #[B, L * c, h, w]  

            
            mid_time = time.time()
            assert not torch.isnan(depth_latent).any(), f"NaN in depth_latent at epoch {epoch} batch {batch_index}"
            #2. process the depth latent to get depth condition for noise prediction
            depth_cond = model("depth_latent_processor", depth_latent=depth_latent) #[B, 256]
            assert not torch.isnan(depth_cond).any(), f"NaN in depth_cond after depth_latent_processor at epoch {epoch} batch {batch_index}"

            depth_cond = model("deep_floor_net", floor_plan=floor_plan.to(rank), cur_pos=cur_pos.to(rank), cur_heading=cur_heading.to(rank), goal_pos=metric_goal, depth_cond=depth_cond)
            assert not torch.isnan(depth_cond).any(), f"NaN in depth_cond after deep_floor_net at epoch {epoch} batch {batch_index}"
            #3. add noise to shortest actions to get noisy global direction condition for noise prediction
            B = shortest_actions.shape[0]
            shortest_actions = shortest_actions.to(rank)
            # shortest_actions = add_sin_noise(epoch, epochs, max_sigma, shortest_actions).float().to(rank)
            #4. add nosie to collision free actions to serve as input for noise prediction
            deltas = get_delta(collision_free_actions.numpy())
            ndeltas = normalize_data(deltas, ACTION_STATS)
            naction = from_numpy(ndeltas).to(rank)
            assert naction.shape[-1] == 2, "action dim must be 2"
            assert not torch.isnan(naction).any(), f"NaN in naction at epoch {epoch} batch {batch_index}"
            noise = torch.randn(naction.shape).to(rank)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (B,)
            ).long().to(rank)
            noisy_action = noise_scheduler.add_noise(
                naction, noise, timesteps)
            #5. predict the noise residual and reverse the diffusion process to get the action

            noise_pred = model("noise_pred_net", sample=noisy_action, timestep=timesteps, global_cond=depth_cond, local_cond=shortest_actions)
            assert not torch.isnan(noise_pred).any(), f"NaN in noise_pred at epoch {epoch} batch {batch_index}"

            diffusion_loss = action_reduce(F.mse_loss(noise_pred, noise, reduction="none"))
            loss = diffusion_loss
            
            #6. update the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
            torch.cuda.empty_cache()

            #Logging
            loss_item = loss.item()
            print(f"[epoch {epoch} batch {batch_index}] loss = {loss_item}")
            tepoch.set_postfix(loss=loss_item)
            end_time = time.time()
            if rank == 0:
                step = epoch * num_batches + batch_index
                if writter is not None:
                    writter.add_scalar('train/loss', loss_item, step)
                if logger == "wandb":
                    import wandb
                    wandb.log({'train/loss': loss_item}, step=step)
            
            if batch_index % print_log_freq == 0 :
                losses = _compute_losses(
                            model,
                            noise_scheduler,
                            depth_cond,
                            shortest_actions,
                            collision_free_actions,
                            # metric_goal,
                            device,
                        )
              
                for key, value in losses.items():
                    if key in loggers:
                        metric_logger = loggers[key]
                        metric_logger.log_data(value.item())

                data_log = {}
                step = epoch * num_batches + batch_index
                for key, metric_logger in loggers.items():
                    name = metric_logger.full_name()
                    data_log[name] = from_numpy(np.array(metric_logger.latest())).to(rank)
                    dist.all_reduce(data_log[name], op=dist.ReduceOp.SUM)
                    data_log[name] /= dist.get_world_size()

                if rank == 0:
                    scalars = {f'train/{name}': value.item() for name, value in data_log.items()}
                    scalars['Total time per batch'] = end_time - start_time
                    scalars['Generate time per batch'] = mid_time - start_time
                    if writter is not None:
                        for scalar_name, scalar_val in scalars.items():
                            writter.add_scalar(scalar_name, scalar_val, step)
                    if logger == "wandb":
                        import wandb
                        wandb.log(scalars, step=step)

            if image_log_freq > 0 and batch_index % image_log_freq == 0 and rank == 0:
                step = epoch * num_batches + batch_index
                fig = visualize_trajectory(
                    model, noise_scheduler, depth_cond, shortest_actions,
                    collision_free_actions.to(rank),
                    num_images=num_images_log, rank=rank,
                )
                if writter is not None:
                    writter.add_figure('train/trajectory', fig, step)
                if logger == "wandb":
                    import wandb
                    wandb.log({'train/trajectory': wandb.Image(fig)}, step=step)
                plt.close(fig)



# def execute_model(
#     ema_model: EMAModel,
#     cur_pos: np.ndarray,        # np.array (1,2)   
#     cur_heading: np.ndarray,    # np.array (1,2)
#     # cur_pos_f3: np.ndarray, # np.array (1,3)      
#     # cur_heading_f3: np.ndarray, # np.array (1,3)  
#     goal_pos: np.ndarray,       # np.array (1,2)
#     # img_paths: List[str],       # list of image paths
#     # floorplan_path: str,        # floorplan path
#     cur_obs: torch.Tensor,        # torch.tensor(L,3,h,w)
#     floorplan: torch.Tensor,      # torch.tensor (1,3,h,w)
#     metric_waipoint_spacing: float,
#     waypoint_spacing: float,
#     transform: transforms,
#     device: torch.device,
#     noise_scheduler: DDPMScheduler,
#     floorplan_ary: np.ndarray,
#     log_add: str = None,
# ):
#     """
#     Execute the model on the given data.
#     Args:
#         ema_model: exponential moving average model
#         cur_pos: current position
#         goal_pos: goal position
#         # img_paths: list of image paths
#         # floorplan_path: floorplan path
#         cur_obs: current observation
#         floorplan: floorplan
#         transform: transform to apply to images
#         device: device to use for evaluation
#         noise_scheduler: noise scheduler to evaluate with 
#         log_folder: folder to save images to
#     """
#     ema_model = ema_model.averaged_model
#     ema_model.eval()
#     
#     cur_pos = torch.as_tensor(cur_pos, dtype=torch.float32)
#     cur_heading = torch.as_tensor(cur_heading, dtype=torch.float32)
#     goal_pos = torch.as_tensor(goal_pos, dtype=torch.float32)
#     cur_obs = torch.as_tensor(cur_obs, dtype=torch.float32)
#     floorplan = torch.as_tensor(floorplan, dtype=torch.float32)
#     cur_pos /= metric_waipoint_spacing * waypoint_spacing
#     cur_heading /= metric_waipoint_spacing * waypoint_spacing
#     goal_pos /= metric_waipoint_spacing * waypoint_spacing
#     cur_obss = torch.split(cur_obs, 1, dim=0)
#     batch_cur_obss = [transform(obs) for obs in cur_obss]
#     batch_cur_obss = torch.cat(batch_cur_obss, dim=1).to(device)  # (1,3*L,h,w)
#     batch_floorplan = transform(floorplan).to(device)  # (1,3,h,w)
#     
#     # cur_pos_f3 = torch.as_tensor(cur_pos_f3, dtype=torch.float32)
#     # cur_heading_f3 = torch.as_tensor(cur_heading_f3, dtype=torch.float32)
#     
#     # _, cur_pos_resized, goal_pos_resized, cur_ori_resized = img_path_to_data_and_point_transfer('/home/user/data/vis_nav/iGibson/igibson/dataset/Quantico_220/train/Quantico/traj_127/00072.png', (96, 96), cur_pos[0], goal_pos[0], cur_heading[0])
#     # cur_pos_i = torch.tensor(np.array([cur_pos_resized]))
#     # goal_pos_i = torch.tensor(np.array([goal_pos_resized]))
#     # cur_heading_i = torch.tensor(np.array([cur_ori_resized]))
#     
#     model_output_dict = model_output(
#         ema_model,
#         noise_scheduler,
#         batch_cur_obss,
#         batch_floorplan,
#         32,
#         2,
#         30,
#         goal_pos,
#         cur_pos,
#         cur_heading,
#         device=device,
#     )
#     actions = model_output_dict['actions'].mean(dim=0)  # [1,8,2]
#     
#     # actions = actions.squeeze(0)
#     distance = model_output_dict['distance']
#     # pos_ori = model_output_dict['pos_ori']
#     # print('pos_ori shape ', pos_ori.shape)
#     # print(pos_ori.mean(dim=0))
#     # print('gt ', cur_pos, cur_heading )
#     actions_normed_global = to_global_coords(to_numpy(actions), to_numpy(cur_pos).squeeze(0), to_numpy(cur_heading).squeeze(0))
#     actions_meter_global = actions_normed_global * metric_waipoint_spacing * waypoint_spacing
#     
#     
#     # if f3
#     # actions_meter_global_transformed = actions_meter_global.copy()
#     # for i in range(actions_meter_global.shape[0]):
#     #     actions_meter_global_transformed[i] = actions_meter_global[i] - actions_meter_global[0] + cur_pos.squeeze(0)
#     
#     if log_add is not None:
#         save_action = actions.cpu().detach().numpy()
#         gs = gridspec.GridSpec(6, 6)
#         gs.update(wspace = 0.9, hspace = 0.7)
#         ax1 = plt.subplot(gs[:2, :2])
#         ax2 = plt.subplot(gs[:2, 2:])
#         ax3 = plt.subplot(gs[2:, :3])
#         ax4 = plt.subplot(gs[2:, 3:])
#         
#         goal_pos_metric = goal_pos * metric_waipoint_spacing * waypoint_spacing
#         floor_width = floorplan_ary.shape[0]
#         end_xy = np.flip((np.array(goal_pos_metric[0]) / 0.01 + floor_width / 2.0)).astype(int)
#         start_xy = np.flip((np.array(goal_pos_metric[0]) / 0.01 + floor_width / 2.0)).astype(int)
#         floorplan_ary[max(0, end_xy[0]-5) : min(end_xy[0]+5, floorplan_ary.shape[0]), max(0, end_xy[1]-5) : min(end_xy[1]+5, floorplan_ary.shape[1]), :] = np.array([0, 0, 255, 255])
#         
#         ax1.imshow(cur_obs[-1].permute(1,2,0).cpu().detach().numpy())
#         ax2.plot(save_action[:,0], save_action[:,1], marker = '.')
#         for i, xy in enumerate(actions_meter_global):
#             map_xy = np.flip((np.array(xy) / 0.01 + floor_width / 2.0)).astype(int)
#             if i == 0:
#                 start_xy = map_xy
#             if i < 8:
#                 color = np.array([255, 0, 0, 255])
#                 floorplan_ary[map_xy[0]-2 : map_xy[0]+2, map_xy[1]-2 : map_xy[1]+2, :] = color
#             else:
#                 color = np.array([0, 255, 0, 30])
#                 floorplan_ary[map_xy[0]-1 : map_xy[0]+1, map_xy[1]-1 : map_xy[1]+1, :] = color
#             
#         ax3.imshow(floorplan_ary[max(0, start_xy[0]-200) : min(start_xy[0]+200, floorplan_ary.shape[0]), max(0, start_xy[1]-200) : min(start_xy[1]+200, floorplan_ary.shape[1]), :])
#         ax4.imshow(floorplan_ary)
#         # plt.plot(actions_meter_global[:,0], actions_meter_global[:,1], marker = 'o')
#         plt.savefig(os.path.join(log_add))
#     
#     return actions_meter_global    
#     


# # normalize data
# def get_data_stats(data):
#     data = data.reshape(-1,data.shape[-1])
#     stats = {
#         'min': np.min(data, axis=0),
#         'max': np.max(data, axis=0)
#     }
#     return stats
# 
def normalize_data(data, stats):
    """
    Normalize data to [-1, 1] using per-dimension min/max statistics.

    Args:
        data (np.ndarray): data to normalize
        stats (dict): dict with 'min' and 'max' arrays
    Returns:
        np.ndarray: normalized data
    """
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    """
    Invert normalize_data, mapping [-1, 1] back to the original data range.

    Args:
        ndata (np.ndarray): normalized data
        stats (dict): dict with 'min' and 'max' arrays
    Returns:
        np.ndarray: data in the original range
    """
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

def get_delta(actions):       # (0,0)->first action point, first action point->second action point, ...
    """
    Convert a sequence of absolute waypoints into step-to-step deltas.

    Args:
        actions (np.ndarray): absolute waypoints, shape (B, T, action_dim)
    Returns:
        np.ndarray: deltas between consecutive waypoints (first delta is from the origin), shape (B, T, action_dim)
    """
    # append zeros to first action
    ex_actions = np.concatenate([np.zeros((actions.shape[0],1,actions.shape[-1])), actions], axis=1)
    delta = ex_actions[:,1:] - ex_actions[:,:-1]
    return delta

def get_action(diffusion_output, action_stats=ACTION_STATS):
    """
    Convert a diffusion model's predicted normalized deltas into absolute waypoints.

    Args:
        diffusion_output (torch.Tensor): predicted normalized deltas, shape (B, T, 2)
        action_stats (dict): dict with 'min' and 'max' arrays used to unnormalize
    Returns:
        torch.Tensor: absolute waypoints, shape (B, T, 2)
    """
    # diffusion_output: (B, 2*T+1, 1)
    # return: (B, T-1)
    device = diffusion_output.device
    ndeltas = diffusion_output
    ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
    ndeltas = to_numpy(ndeltas)
    ndeltas = unnormalize_data(ndeltas, action_stats)
    actions = np.cumsum(ndeltas, axis=1)
    return from_numpy(actions).to(device)


def model_output(
    model: nn.Module,
    noise_scheduler: DDPMScheduler,
    depth_cond: torch.Tensor,
    shortest_actions: torch.Tensor,
    pred_horizon: int,
    action_dim: int,  
    # metric_goal: torch.Tensor,   
    device: torch.device,
):
    """
    Run the full DDPM denoising loop to predict actions from a condition embedding.

    Args:
        model (nn.Module): glocdiff model
        noise_scheduler (DDPMScheduler): diffusion noise scheduler
        depth_cond (torch.Tensor): fused condition embedding, shape (B, encoding_size)
        shortest_actions (torch.Tensor): local-condition actions, shape (B, len_traj_pred, 2)
        pred_horizon (int): number of waypoints to predict
        action_dim (int): dimensionality of each waypoint
        device (torch.device): device to run inference on
    Returns:
        dict: predicted 'actions', shape (B, pred_horizon, action_dim)
    """
    rank = int(os.environ['LOCAL_RANK'])

    # initialize action from Gaussian noise
    noisy_diffusion_output = torch.randn((depth_cond.shape[0], pred_horizon, action_dim)).to(rank)
    diffusion_output = noisy_diffusion_output
 
    for k in noise_scheduler.timesteps[:]:
        # predict noise

        noise_pred = model(
            "noise_pred_net",
            sample=diffusion_output,
            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(rank),
            global_cond=depth_cond,
            local_cond=shortest_actions
            # local_cond=metric_goal
        )
        noise_pred = noise_pred.to(rank)

        # inverse diffusion step (remove noise)
        diffusion_output = noise_scheduler.step(
            model_output=noise_pred,
            timestep=int(k),
            sample=diffusion_output
        ).prev_sample

    actions = get_action(diffusion_output, ACTION_STATS)  

    return {
        'actions': actions
    }


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Detach a tensor from the graph, move it to CPU, and convert it to a NumPy array.

    Args:
        tensor (torch.Tensor): input tensor
    Returns:
        np.ndarray: detached CPU array
    """
    return tensor.detach().cpu().numpy()


def from_numpy(array: np.ndarray) -> torch.Tensor:
    """
    Convert a NumPy array to a float32 torch tensor.

    Args:
        array (np.ndarray): input array
    Returns:
        torch.Tensor: float32 tensor
    """
    return torch.from_numpy(array).float()

# def visualize_diffusion_action_distribution(
#     ema_model: nn.Module,
#     noise_scheduler: DDPMScheduler,
#     batch_obs_images: torch.Tensor,
# 
#     batch_viz_obs_images: torch.Tensor,
#     batch_viz_goal_images: torch.Tensor,
#     batch_action_label: torch.Tensor,
#     batch_distance_labels: torch.Tensor,
#     batch_goal_pos: torch.Tensor,
#     batch_curr_pos: torch.Tensor,
#     batch_curr_ori: torch.Tensor,
#     batch_goal_pos_local: torch.Tensor,
#     batch_goal_pos_resized: torch.Tensor,
#     batch_curr_pos_resized: torch.Tensor,
#     batch_curr_ori_resized: torch.Tensor,
#     
#     device: torch.device,
#     type: str,
#     project_folder: str,
#     epoch: int,
#     num_images_log: int,
#     num_samples: int = 30,
#     use_wandb: bool = True,
# ):
#     """Plot samples from the exploration model."""
# 
#     visualize_path = os.path.join(
#         project_folder,
#         "visualize",
#         type,
#         f"epoch{epoch}",
#         "action_sampling_prediction",
#     )
#     if not os.path.isdir(visualize_path):
#         os.makedirs(visualize_path)
# 
#     max_batch_size = batch_obs_images.shape[0]
# 
#     num_images_log = min(num_images_log, batch_obs_images.shape[0], batch_goal_images.shape[0], batch_action_label.shape[0], batch_goal_pos.shape[0])
#     batch_obs_images = batch_obs_images[:num_images_log]
#     batch_goal_images = batch_goal_images[:num_images_log]
#     batch_action_label = batch_action_label[:num_images_log]
#     wandb_list = []
# 
#     pred_horizon = batch_action_label.shape[1]
#     action_dim = batch_action_label.shape[2]
# 
#     # split into batches
#     batch_obs_images_list = torch.split(batch_obs_images, max_batch_size, dim=0)
#     actions_list = []
# 
#     for depth_image, shortest_actions in zip(batch_obs_images_list):
#         model_output_dict = model_output(
#             ema_model,
#             noise_scheduler,
#             obs,
#             pred_horizon,
#             action_dim,
#             num_samples,
#             device,
#         )
#         actions_list.append(to_numpy(model_output_dict['actions'])) # local, waypoints metric
# 
#     # concatenate
#     actions_list = np.concatenate(actions_list, axis=0)
# 
#     # split into actions per observation
#     actions_list = np.split(actions_list, num_images_log, axis=0)
# 
#     distances_avg = [np.mean(dist) for dist in distances_list]
#     distances_std = [np.std(dist) for dist in distances_list]
# 
#     assert len(actions_list) == len(actions_list) == num_images_log
# 
#     np_distance_labels = to_numpy(batch_distance_labels)
# 
#     for i in range(num_images_log):
#         fig, ax = plt.subplots(1, 3)
#         actions = actions_list[i]
#         action_label = to_numpy(batch_action_label[i])
# 
#         traj_list = np.concatenate([
#             actions,
#             action_label[None],
#         ], axis=0)
#         # print("traj_list.shape", traj_list.shape)   
#         # traj_labels = ["r", "GC", "GC_mean", "GT"]
#         traj_colors = ["red"] * len(actions) + ["magenta"]
#         traj_alphas = [0.1] * len(actions) + [1.0]
# 
#         # make points numpy array of robot positions (0, 0) and goal positions
#         # point_list = [np.array([0, 0]), to_numpy(batch_goal_pos[i])]
#         point_list = [np.array([0, 0]), to_numpy(batch_goal_pos_local[i])]
#         point_colors = ["green", "red"]
#         point_alphas = [1.0, 1.0]
# 
#         plot_trajs_and_points(
#             ax[0],
#             traj_list,
#             point_list,
#             traj_colors,
#             point_colors,
#             traj_labels=None,
#             point_labels=None,
#             quiver_freq=0,
#             traj_alphas=traj_alphas,
#             point_alphas=point_alphas, 
#         )
#         
#         obs_image = to_numpy(batch_viz_obs_images[i])
#         goal_image = to_numpy(batch_viz_goal_images[i])
#         # move channel to last dimension
#         obs_image = np.moveaxis(obs_image, 0, -1)
#         goal_image = np.moveaxis(goal_image, 0, -1)
#         ax[1].imshow(obs_image)
#         ax[2].imshow(goal_image)
# 
#         # set title
#         ax[0].set_title(f"diffusion action predictions")
#         ax[1].set_title(f"observation")
#         ax[2].set_title(f"goal: label={np_distance_labels[i]} gc_dist={distances_avg[i]:.2f}±{distances_std[i]:.2f}")
#         
#         str_text = f'goal_resized:{batch_goal_pos_resized[i].cpu().numpy()} curr_pos_resized:{batch_curr_pos_resized[i].cpu().numpy()} curr_ori_resized:{batch_curr_ori_resized[i].cpu().numpy()}'
#         fig.text(0, 0, str_text)
#         
#         # make the plot large
#         fig.set_size_inches(18.5, 10.5)
# 
#         save_path = os.path.join(visualize_path, f"sample_{i}.png")
#         plt.savefig(save_path)
#         # wandb_list.append(wandb.Image(save_path))
#         plt.close(fig)
#     if len(wandb_list) > 0 and use_wandb:
#         wandb.log({f"{type}_action_samples": wandb_list}, commit=False)
# 
# def plot_trajs_and_points(
#     ax: plt.Axes,
#     list_trajs: list,
#     list_points: list,
#     traj_colors: list = [CYAN, MAGENTA],
#     point_colors: list = [RED, GREEN],
#     traj_labels: Optional[list] = ["prediction", "ground truth"],
#     point_labels: Optional[list] = ["robot", "goal"],
#     traj_alphas: Optional[list] = None,
#     point_alphas: Optional[list] = None,
#     quiver_freq: int = 1,
#     default_coloring: bool = True,
# ):
#     """
#     Plot trajectories and points that could potentially have a yaw.
# 
#     Args:
#         ax: matplotlib axis
#         list_trajs: list of trajectories, each trajectory is a numpy array of shape (horizon, 2) (if there is no yaw) or (horizon, 4) (if there is yaw)
#         list_points: list of points, each point is a numpy array of shape (2,)
#         traj_colors: list of colors for trajectories
#         point_colors: list of colors for points
#         traj_labels: list of labels for trajectories
#         point_labels: list of labels for points
#         traj_alphas: list of alphas for trajectories
#         point_alphas: list of alphas for points
#         quiver_freq: frequency of quiver plot (if the trajectory data includes the yaw of the robot)
#     """
#     assert (
#         len(list_trajs) <= len(traj_colors) or default_coloring
#     ), "Not enough colors for trajectories"
#     assert len(list_points) <= len(point_colors), "Not enough colors for points"
#     assert (
#         traj_labels is None or len(list_trajs) == len(traj_labels) or default_coloring
#     ), "Not enough labels for trajectories"
#     assert point_labels is None or len(list_points) == len(point_labels), "Not enough labels for points"
# 
#     for i, traj in enumerate(list_trajs):
#         if traj_labels is None:
#             ax.plot(
#                 traj[:, 0], 
#                 traj[:, 1], 
#                 color=traj_colors[i],
#                 alpha=traj_alphas[i] if traj_alphas is not None else 1.0,
#                 marker="o",
#             )
#         else:
#             ax.plot(
#                 traj[:, 0],
#                 traj[:, 1],
#                 color=traj_colors[i],
#                 label=traj_labels[i],
#                 alpha=traj_alphas[i] if traj_alphas is not None else 1.0,
#                 marker="o",
#             )
#         if traj.shape[1] > 2 and quiver_freq > 0:  # traj data also includes yaw of the robot
#             bearings = gen_bearings_from_waypoints(traj)
#             ax.quiver(
#                 traj[::quiver_freq, 0],
#                 traj[::quiver_freq, 1],
#                 bearings[::quiver_freq, 0],
#                 bearings[::quiver_freq, 1],
#                 color=traj_colors[i] * 0.5,
#                 scale=1.0,
#             )
#     for i, pt in enumerate(list_points):
#         if point_labels is None:
#             ax.plot(
#                 pt[0], 
#                 pt[1], 
#                 color=point_colors[i], 
#                 alpha=point_alphas[i] if point_alphas is not None else 1.0,
#                 marker="o",
#                 markersize=7.0
#             )
#         else:
#             ax.plot(
#                 pt[0],
#                 pt[1],
#                 color=point_colors[i],
#                 alpha=point_alphas[i] if point_alphas is not None else 1.0,
#                 marker="o",
#                 markersize=7.0,
#                 label=point_labels[i],
#             )
# 
#     
#     # put the legend below the plot
#     if traj_labels is not None or point_labels is not None:
#         ax.legend()
#         ax.legend(bbox_to_anchor=(0.0, -0.5), loc="upper left", ncol=2)
#     ax.set_aspect("equal", "box")
# 
def add_sin_noise(epoch, epochs, max_sigma, trajetory):
    """
    Add Gaussian noise to a trajectory, with amplitude following a sine
    curriculum that ramps up then back down over training.

    Args:
        epoch (int): current epoch
        epochs (int): total number of epochs
        max_sigma (float): peak noise standard deviation
        trajetory (np.ndarray): trajectory to perturb
    Returns:
        np.ndarray: trajectory with noise added
    """
    sigma_t = max_sigma * np.sin(np.pi * (epoch % epochs) / (2 * epochs))
    trajetory += np.random.normal(0, sigma_t, trajetory.shape)
    return trajetory

# def check_collision(pos, travers_map):
#     """
#     check if the position is in collision
#     """
#     # print(pos)
#     (x, y) = pos * 100 + np.array([travers_map.shape[0], travers_map.shape[1]]) // 2
#     if x < 0 or x > travers_map.shape[1] or y < 0 or y >= travers_map.shape[0]:
#         return True
#     pos_in_map = (pos * 100 + np.array([travers_map.shape[0], travers_map.shape[1]]) // 2).astype(np.int16)
#     return travers_map[pos_in_map[1], pos_in_map[0]] == 0