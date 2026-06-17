import os
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np
import yaml
import time
import sys

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler

import diffusers
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "diffusion_policy"))


from model.glocdiff import glocdiff
from model.depth_latent_processor import depth_latent_processor
from model.deepfloor_net import deep_floor_net
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D


from glocdiff_dataset import GlocDiffDataset
from train_utils import train_one_epoch

def load_model(model, checkpoint: dict) -> None:
    """
    Load a model's weights from a checkpoint state dict.

    Args:
        model (nn.Module): model to load weights into
        checkpoint (dict): state dict to load
    """

    state_dict = checkpoint
    # if 'module' in list(state_dict.keys())[0]:
    #     state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

def train_gloc(
    model: nn.Module,
    optimizer: Adam, 
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    noise_scheduler: DDPMScheduler,
    train_loader: DataLoader,
    epochs: int,
    data_folder: str,
    device: torch.device,
    ckpt_folder: str,
    print_log_freq: int = 100,
    image_log_freq: int = 50,
    num_images_log: int = 8,
    current_epoch: int = 0,
    alpha: float = 1e-4,
    beta: float = 1,
    logger: str = "none",
    max_sigma: float = 0.01,
    writter=None,
    metric_waypoint_spacing: float = 0.045,
    waypoint_spacing: float = 1,
    # dictpath: str=None  
):
    """
    Train the model for several epochs, saving a checkpoint after each one.

    Args:
        model (nn.Module): model to train
        optimizer (Adam): optimizer to use
        lr_scheduler: learning rate scheduler to use
        noise_scheduler (DDPMScheduler): diffusion noise scheduler
        train_loader (DataLoader): dataloader for the training dataset
        epochs (int): total number of epochs to train through
        data_folder (str): root folder containing the scene data and depth latent cache
        device (torch.device): device to train on
        ckpt_folder (str): folder to save checkpoints to
        print_log_freq (int): how often (in batches) to log scalar metrics
        image_log_freq (int): how often (in batches) to log trajectory visualizations
        num_images_log (int): number of samples to visualize per trajectory plot
        current_epoch (int): epoch to start training from (for resuming)
        alpha (float): weight of the action loss
        beta (float): weight of the distance loss
        logger (str): which logging backend to use ("tensorboard", "wandb", or "none")
        max_sigma (float): maximum noise amplitude for the local-condition curriculum noise
        writter: TensorBoard SummaryWriter, or None if not using TensorBoard
        metric_waypoint_spacing (float): meters per waypoint unit
        waypoint_spacing (int): number of frames between consecutive waypoints
    """
    rank = int(os.environ['LOCAL_RANK'])
    depth_generator = diffusers.MarigoldDepthPipeline.from_pretrained(
        "prs-eth/marigold-depth-lcm-v1-0", variant="fp16", torch_dtype=torch.float16
    ).to(rank)
    for epoch in range(current_epoch, epochs):
        train_loader.sampler.set_epoch(epoch)
        if rank == 0:
            print(
            f"Start Training Epoch {epoch}/{epochs - 1}"
            )
        start_time = time.time()
        train_one_epoch(
            model=model,
            optimizer=optimizer,
            dataloader=train_loader,
            device=device,
            noise_scheduler=noise_scheduler,
            depth_generator=depth_generator,
            epoch=epoch,
            data_folder=data_folder,
            alpha=alpha,
            beta=beta,
            print_log_freq=print_log_freq,
            image_log_freq=image_log_freq,
            num_images_log=num_images_log,
            logger=logger,
            epochs=epochs,  
            max_sigma=max_sigma,
            writter=writter,    
            metric_waypoint_spacing=metric_waypoint_spacing,
            waypoint_spacing=waypoint_spacing,
            # dictpath=dictpath
        )
        end_time = time.time()
        lr_scheduler.step()
        if rank == 0:
            time_for_epoch = end_time - start_time
            print(f"Time for epoch {epoch}: {time_for_epoch}")

        latest_path = os.path.join(ckpt_folder, "latest.pth")
        

        numbered_path = os.path.join(ckpt_folder, f"{epoch}.pth")
        torch.save(model.state_dict(), numbered_path)
        torch.save(model.state_dict(), latest_path)
        with open(os.path.join(ckpt_folder, "latest_epoch.txt"), "w") as f:
            f.write(str(epoch))

        # save optimizer
        latest_optimizer_path = os.path.join(ckpt_folder, "optimizer_latest.pth")
        torch.save(optimizer.state_dict(), latest_optimizer_path)

        # save scheduler
        latest_scheduler_path = os.path.join(ckpt_folder, "scheduler_latest.pth")
        torch.save(lr_scheduler.state_dict(), latest_scheduler_path)


        if rank == 0:
            lr = optimizer.param_groups[0]["lr"]
            if writter is not None:
                writter.add_scalar("learning_rate", lr, epoch)
            if logger == "wandb":
                import wandb
                wandb.log({"epoch": epoch, "learning_rate": lr})
        # lr_scheduler.step()
 


def main(config):
    """
    Set up distributed training, build the model/dataset/optimizer, optionally
    resume from a checkpoint, and run the training loop.

    Args:
        config (dict): parsed training configuration (see config/glocdiff.yaml)
    """
    torch.cuda.empty_cache()

    
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')  
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    
    writter = None
    if local_rank == 0:
        if config["logger"] == "wandb":
            import wandb
            wandb.init(project=config["project_name"], name=config["run_name"])
            wandb.define_metric("learning_rate", step_metric="epoch")
        elif config["logger"] == "tensorboard":
            tb_log_dir = os.path.join("runs", config["project_name"], config["run_name"])
            writter = SummaryWriter(log_dir=tb_log_dir)
            
    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True

    cudnn.benchmark = True  # good if input sizes don't vary


    data_config = config["datasets"]

    train_dataset = GlocDiffDataset(
        data_folder=os.path.join(data_config["data_folder"], "train"),
        scene_names=data_config['scenes_for_training'],
        image_size=config["image_size"],
        waypoint_spacing=data_config["waypoint_spacing"],
        len_traj_pred=config["len_traj_pred"],
        context_size=config["context_size"],
        end_slack=data_config["end_slack"],
        normalize=config["normalize"],
        obs_type=config["obs_type"],
        trav_map_path=data_config["traversable_map_folder"],
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        drop_last=False,
        persistent_workers=False,
        sampler=train_sampler,  
    )

    deepfloor_net = deep_floor_net(
        floorplan_encoder=config["floorplan_encoder"],
        floorplan_encoding_size=config["encoding_size"],
    )

    depth_processor = depth_latent_processor(
        depth_encoding_size=config["encoding_size"],
        context_size=config["context_size"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
    )
    # rgbprocesser = rgb_processer(
    #     context_size=config["context_size"],
    #     rgb_encoding_size=config["encoding_size"],
    #     mha_num_attention_heads=config["mha_num_attention_heads"],
    #     mha_num_attention_layers=config["mha_num_attention_layers"],
    #     mha_ff_dim_factor=config["mha_ff_dim_factor"],
    # )
    
    noise_pred_net = ConditionalUnet1D(
            input_dim=2,
            local_cond_dim=2,
            global_cond_dim=config["encoding_size"],    # +6
            down_dims=config["down_dims"],
            cond_predict_scale=config["cond_predict_scale"],
        )   
    model = glocdiff(
        depth_latent_processor=depth_processor,
        noise_pred_net=noise_pred_net,
        deep_floor_net=deepfloor_net,
    ).to(local_rank)

    model = nn.parallel.DistributedDataParallel(model, device_ids=None, output_device=None, find_unused_parameters=True)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_diffusion_iters"],
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )


    lr = float(config["lr"])
    config["optimizer"] = config["optimizer"].lower()
    if config["optimizer"] == "adam":
        optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    elif config["optimizer"] == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr)
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not supported")

    scheduler = None
    if config["scheduler"] is not None:
        config["scheduler"] = config["scheduler"].lower()
        if config["scheduler"] == "cosine" :
            if local_rank == 0:
                print("Using cosine annealing with T_max", config["epochs"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["epochs"]
            )
        elif config["scheduler"] == "cyclic":
            print("Using cyclic LR with cycle", config["cyclic_period"])
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=lr / 10.,
                max_lr=lr,
                step_size_up=config["cyclic_period"] // 2,
                cycle_momentum=False,
            )
        elif config["scheduler"] == "plateau":
            print("Using ReduceLROnPlateau")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=config["plateau_factor"],
                patience=config["plateau_patience"],
                verbose=True,
            )
        else:
            raise ValueError(f"Scheduler {config['scheduler']} not supported")

        if config["warmup"]:
            if local_rank == 0:
                print("Using warmup scheduler")
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=config["warmup_epochs"],
                after_scheduler=scheduler,
            )

    current_epoch = 0 

    if config.get("load_run") is not None:
        load_run = config["load_run"]
        if not os.path.isabs(load_run):
            load_run = os.path.join("logs", load_run)

        if os.path.isfile(load_run):
            # A single standalone weights file: load weights only, start fresh otherwise
            print("Loading model weights from ", load_run)
            latest_checkpoint = torch.load(load_run)
            load_model(model, latest_checkpoint)
        else:
            load_ckpt_folder = load_run
            print("Loading model from ", load_ckpt_folder)
            latest_path = os.path.join(load_ckpt_folder, "latest.pth")
            latest_checkpoint = torch.load(latest_path) #f"cuda:{}" if torch.cuda.is_available() else "cpu")
            load_model(model, latest_checkpoint)
            epoch_file = os.path.join(load_ckpt_folder, "latest_epoch.txt")
            if os.path.exists(epoch_file):
                current_epoch = int(open(epoch_file).read().strip()) + 1
            else:
                numbered_epochs = [
                    int(f[:-4]) for f in os.listdir(load_ckpt_folder)
                    if f.endswith(".pth") and f[:-4].isdigit()
                ]
                current_epoch = max(numbered_epochs) + 1 if numbered_epochs else 0
            print(f"Resuming from epoch {current_epoch}")
            #load optimizer
            latest_optimizer_path = os.path.join(load_ckpt_folder, "optimizer_latest.pth")
            optimizer.load_state_dict(torch.load(latest_optimizer_path))
            #load scheduler
            latest_scheduler_path = os.path.join(load_ckpt_folder, "scheduler_latest.pth")
            scheduler.load_state_dict(torch.load(latest_scheduler_path))



    train_gloc(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        noise_scheduler=noise_scheduler,
        train_loader=train_loader,
        epochs=config["epochs"],
        data_folder=data_config["data_folder"],
        device=device,
        ckpt_folder=config["ckpt_folder"],
        print_log_freq=config["print_log_freq"],
        image_log_freq=config["image_log_freq"],
        num_images_log=config["num_images_log"],
        current_epoch=current_epoch,
        alpha=float(config["alpha"]),
        beta=float(config["beta"]), 
        logger=config["logger"],
        max_sigma=config["max_sigma"], 
        writter=writter, 
        metric_waypoint_spacing=config["metric_waypoint_spacing"],
        waypoint_spacing=data_config["waypoint_spacing"],    
        # dictpath=config["dictpath"] 
    )
    dist.destroy_process_group()
    print("FINISHED TRAINING")
    


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Visual Navigation Transformer")

    # project setup
    parser.add_argument(
        "--config",
        "-c",
        default="/home/weiqi/code/GlocDiff/config/glocdiff.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )
    args = parser.parse_args()



    with open(args.config, "r") as f:
        config = yaml.safe_load(f)


    config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    config["ckpt_folder"] = os.path.join(
        "logs", config["project_name"], config["run_name"]
    )
    
    if not os.path.exists(config["ckpt_folder"]):
        os.makedirs(config["ckpt_folder"])


    main(config)
