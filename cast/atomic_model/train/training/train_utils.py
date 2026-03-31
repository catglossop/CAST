import wandb
import os
import numpy as np
import yaml
from typing import Callable
import tqdm
import itertools
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

from cast.atomic_model.train.visualizing.action_utils import visualize_traj_pred
from cast.atomic_model.train.visualizing.lang_utils import visualize_lang_pred
from cast.atomic_model.train.visualizing.visualize_utils import to_numpy, from_numpy
from cast.atomic_model.train.training.logger import Logger
from cast.atomic_model.utils.data_utils import VISUALIZATION_IMAGE_SIZE

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

# LOAD DATA CONFIG
with open(os.path.join(os.path.dirname(__file__), "../../configs/data_config.yaml"), "r") as f:
    data_config = yaml.safe_load(f)
# POPULATE ACTION STATS
ACTION_STATS = {}
for key in data_config['action_stats']:
    ACTION_STATS[key] = np.array(data_config['action_stats'][key])

# Train utils for RFT
def _compute_losses(
    action_label: torch.Tensor,
    action_pred: torch.Tensor,
    action_mask: torch.Tensor = None,
):
    """
    Compute losses for distance and action prediction.

    Args:
        action_label: ground truth action
        action_pred: predicted action
        alpha: weight of distance loss
        learn_angle: whether to learn the angle of the action
        action_mask: mask for action prediction
    """

    def action_reduce(unreduced_loss: torch.Tensor):
        # Reduce over non-batch dimensions to get loss per batch element
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
        return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

    assert action_pred.shape == action_label.shape, f"{action_pred.shape} != {action_label.shape}"
    action_loss = action_reduce(F.mse_loss(action_pred, action_label, reduction="none"))

    action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
        action_pred[:, :, :2], action_label[:, :, :2], dim=-1
    ))
    multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
        torch.flatten(action_pred[:, :, :2], start_dim=1),
        torch.flatten(action_label[:, :, :2], start_dim=1),
        dim=-1,
    ))

    results = {
        "action_loss": action_loss,
        "action_waypts_cos_sim": action_waypts_cos_similairity,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim,
    }

    total_loss = action_loss
    results["total_loss"] = total_loss

    return results

def _log_data(
    i,
    epoch,
    num_batches,
    normalized,
    project_folder,
    num_images_log,
    loggers,
    obs_image,
    lang,
    action_pred,
    action_label,
    dataset_names,
    use_wandb,
    mode,
    use_latest,
    wandb_log_freq=1,
    print_log_freq=1,
    image_log_freq=1,
    wandb_increment_step=True,
    metric_waypoint_spacing: float = 0.12,
):
    """
    Log data to wandb and print to console.
    """
    data_log = {}
    for key, logger in loggers.items():
        if use_latest:
            data_log[logger.full_name()] = logger.latest()
            if i % print_log_freq == 0 and print_log_freq != 0:
                print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")
        else:
            data_log[logger.full_name()] = logger.average()
            if i % print_log_freq == 0 and print_log_freq != 0:
                print(f"(epoch {epoch}) {logger.full_name()} {logger.average()}")

    if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
        wandb.log(data_log, commit=wandb_increment_step)

    if image_log_freq != 0 and i % image_log_freq == 0:
        visualize_lang_pred(
            to_numpy(obs_image),
            lang,
            mode,
            project_folder,
            epoch,
            num_images_log,
            use_wandb=use_wandb,
        )
        visualize_traj_pred(
            to_numpy(obs_image),
            list(dataset_names),
            to_numpy(action_pred),
            to_numpy(action_label),
            lang,
            mode,
            normalized,
            project_folder,
            epoch,
            num_images_log,
            use_wandb=use_wandb,
            metric_waypoint_spacing=metric_waypoint_spacing,
        )


def train(
    model: nn.Module,
    ema_model: EMAModel,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
    project_folder: str,
    epoch: int,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    model.train()
    num_batches = len(dataloader)

    total_loss_logger = Logger("total_loss", "train", window_size=print_log_freq)
    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)
    action_waypts_cos_sim_logger = Logger(
        "action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    multi_action_waypts_cos_sim_logger = Logger(
        "multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    loggers = {
        "total_loss": total_loss_logger,
        "action_loss": action_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim_logger,
    }
    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image,
                action_label,
                lang_embedding,
                lang,
                dataset_name,
                metric_waypoint_spacing,
            ) = data
            obs_images = torch.split(obs_image, 3, dim=1)
            viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE)

            obs_images = [transform(obs_image).to(device) for obs_image in obs_images]  
            batch_obs_images = torch.cat(obs_images, dim=1)
            batch_obs_images = batch_obs_images.to(device)
            batch_action_label = action_label.to(device)
            lang_embedding = lang_embedding.to(device)
            action_mask = torch.ones(obs_image.size(0), device=device)

            B = obs_image.size(0)
            obs_cond = model("vision_encoder", obs_img=batch_obs_images, lang_embed=lang_embedding)

            deltas = get_delta(action_label)
            ndeltas = normalize_data(deltas, ACTION_STATS)
            naction = from_numpy(ndeltas).to(device)

            # Predict actions
            eval_model = model.to(device).eval()
            model_outputs = model_output(
                eval_model,
                noise_scheduler,
                batch_obs_images,
                lang_embedding,
                action_label.shape[1],
                2,
                1,
                obs_image.size(0),
                device,
            )
            action_pred = model_outputs["actions"]
            
            losses = _compute_losses(
                action_label=batch_action_label,
                action_pred=action_pred,
                action_mask=action_mask,
            ) 

            # Sample noise to add to actions
            noise = torch.randn(naction.shape, device=device)

            # Sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (B,), device=device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each diffusion iteration
            noisy_action = noise_scheduler.add_noise(
                naction, noise, timesteps)        
                        
            # Predict the noise residual
            noise_pred = model("action_head", sample=noisy_action, timestep=timesteps, global_cond=obs_cond)

            def action_reduce(unreduced_loss: torch.Tensor):
                # Reduce over non-batch dimensions to get loss per batch element
                while unreduced_loss.dim() > 1:
                    unreduced_loss = unreduced_loss.mean(dim=-1)
                assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
                return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

            # L2 loss
            diffusion_loss = action_reduce(F.mse_loss(noise_pred, noise, reduction="none"))

            # Combine losses
            loss = diffusion_loss

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses["total_loss"] = loss

            # Update Exponential Moving Average of the model weights
            ema_model.step(model)

            for key, value in losses.items():
                if key in loggers:
                    logger = loggers[key]
                    logger.log_data(value.item())
            
        # Log data to wandb/console, with visualizations selected from the last batch
        _log_data(
            i=i,
            epoch=epoch,
            num_batches=num_batches,
            normalized=True,
            project_folder=project_folder,
            num_images_log=num_images_log,
            loggers=loggers,
            obs_image=viz_obs_image,
            lang=lang,
            action_pred=action_pred,
            action_label=action_label,
            dataset_names=dataset_name,
            use_wandb=use_wandb,
            mode="train",
            use_latest=False,
            wandb_increment_step=False,
            metric_waypoint_spacing=metric_waypoint_spacing,
        )

        return total_loss_logger.average()
    
def evaluate(
    eval_type: str,
    model: nn.Module,
    ema_model: EMAModel,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
    project_folder: str,
    epoch: int,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    eval_fraction: float = 0.25,
    use_wandb: bool = True,
):
    """
    Evaluate the model on the given evaluation dataset.

    Args:
        eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
        ema_model (nn.Module): exponential moving average version of model to evaluate
        dataloader (DataLoader): dataloader for eval
        transform (transforms): transform to apply to images
        device (torch.device): device to use for evaluation
        noise_scheduler: noise scheduler to evaluate with 
        project_folder (string): path to project folder
        epoch (int): current epoch
        print_log_freq (int): how often to print logs 
        wandb_log_freq (int): how often to log to wandb
        image_log_freq (int): how often to log images
        num_images_log (int): number of images to log
        eval_fraction (float): fraction of data to use for evaluation
        use_wandb (bool): whether to use wandb for logging
    """
    # ema_model = ema_model.averaged_model
    # ema_model.eval()
    model.eval()
    num_batches = len(dataloader)

    action_loss_logger = Logger("action_loss", eval_type, window_size=print_log_freq)
    action_waypts_cos_sim_logger = Logger(
        "action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    multi_action_waypts_cos_sim_logger = Logger(
        "multi_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    diffusion_loss_logger = Logger("diffusion_loss", eval_type, window_size=print_log_freq)
    total_loss_logger = Logger("total_loss", eval_type, window_size=print_log_freq)
    loggers = {
        "action_loss": action_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim_logger,
        "diffusion_loss": diffusion_loss_logger,
        "total_loss": total_loss_logger,
    }
    num_batches = max(int(num_batches * eval_fraction), 1)
    if num_batches > len(dataloader):
        return
    with tqdm.tqdm(
        itertools.islice(dataloader, num_batches), 
        total=num_batches, 
        dynamic_ncols=True, 
        desc=f"Evaluating {eval_type} for epoch {epoch}", 
        leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image,
                action_label,
                lang_embedding,
                lang,
                dataset_name,
                metric_waypoint_spacing,
            ) = data
            
            obs_images = torch.split(obs_image, 3, dim=1)
            viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE)

            obs_images = [transform(obs_image).to(device) for obs_image in obs_images]  
            batch_obs_images = torch.cat(obs_images, dim=1)
            batch_obs_images = batch_obs_images.to(device)
            batch_action_label = action_label.to(device)
            lang_embedding = lang_embedding.to(device)
            action_mask = torch.ones(obs_image.size(0), device=device)
        
            B = obs_image.size(0)
            # CHANGED
            with torch.no_grad():
                obs_cond = model("vision_encoder", obs_img=batch_obs_images, lang_embed=lang_embedding)

                deltas = get_delta(action_label)
                ndeltas = normalize_data(deltas, ACTION_STATS)
                naction = from_numpy(ndeltas).to(device)
                assert naction.shape[-1] == 2, "action dim must be 2"

                model_outputs = model_output(
                    model,
                    noise_scheduler,
                    batch_obs_images,
                    lang_embedding,
                    action_label.shape[1],
                    2,
                    1,
                    B,
                    device,
                )
                action_pred = model_outputs["actions"]

                # Sample noise to add to actions
                noise = torch.randn(naction.shape, device=device)

                # Sample a diffusion iteration for each data point
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (B,), device=device
                ).long()

                noisy_actions = noise_scheduler.add_noise(
                    naction, noise, timesteps)
                # CHANGED
                noise_pred = model("action_head", sample=noisy_actions, timestep=timesteps, global_cond=obs_cond)
                diffusion_loss = nn.functional.mse_loss(noise_pred, noise)

                losses = _compute_losses(
                    action_label=batch_action_label,
                    action_pred=action_pred,
                    action_mask=action_mask,
                )
                loss = diffusion_loss
                losses["diffusion_loss"] = diffusion_loss

                losses["total_loss"] = loss
                
            for key, value in losses.items():
                if key in loggers:
                    logger = loggers[key]
                    logger.log_data(value.item())
            
            data_log = {}
            for key, logger in loggers.items():
                data_log[logger.full_name()] = logger.latest()
                if i % print_log_freq == 0 and print_log_freq != 0:
                    print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)
    
    _log_data(
        i=i,
        epoch=epoch,
        num_batches=num_batches,
        normalized=True,
        project_folder=project_folder,
        num_images_log=num_images_log,
        loggers=loggers,
        obs_image=viz_obs_image,
        lang=lang,
        action_pred=action_pred,
        action_label=action_label,
        dataset_names=dataset_name,
        use_wandb=use_wandb,
        mode=eval_type,
        use_latest=False,
        wandb_increment_step=False,
        metric_waypoint_spacing=metric_waypoint_spacing,
    )

# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # Normalize to [-1, 1].
    # Guard against constant dimensions (max == min), which would otherwise
    # produce NaNs/Infs and can lead to numeric issues later.
    rng = stats["max"] - stats["min"]
    rng = np.where(rng == 0, 1.0, rng)
    ndata = (data - stats["min"]) / rng
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

def get_delta(actions):
    # append zeros to first action
    ex_actions = np.concatenate([np.zeros((actions.shape[0],1,actions.shape[-1])), actions], axis=1)
    delta = ex_actions[:,1:] - ex_actions[:,:-1]
    return delta

def get_action(diffusion_output, action_stats=ACTION_STATS):
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
    batch_obs_images: torch.Tensor,
    batch_lang_embeddings: torch.Tensor,
    pred_horizon: int,
    action_dim: int,
    num_samples: int,
    batch_size: int,
    device: torch.device,
):
    # Stable inference-style diffusion sampling (no gradients).
    model.eval()
    noise_scheduler.set_timesteps(noise_scheduler.config.num_train_timesteps)

    with torch.no_grad():
        obs_cond = model(
            "vision_encoder",
            obs_img=batch_obs_images,
            lang_embed=batch_lang_embeddings,
        )
        obs_cond = obs_cond.reshape(batch_size, -1)
        obs_cond = obs_cond.repeat_interleave(num_samples, dim=0)

        batch_inner = batch_size * num_samples
        diffusion_output = torch.randn(
            batch_inner,
            pred_horizon,
            action_dim,
            device="cpu",
            dtype=torch.float32,
        ).to(device)

        # Optional debug hook to localize where an FPE is triggered.
        # Usage: CAST_MODEL_OUTPUT_DEBUG_STAGE=after_noise
        if os.environ.get("CAST_MODEL_OUTPUT_DEBUG_STAGE", "") == "after_noise":
            return {"actions": diffusion_output}
        for k in noise_scheduler.timesteps:
            noise_pred = model(
                "action_head",
                sample=diffusion_output,
                timestep=k,  # scalar timestep (matches deployment usage)
                global_cond=obs_cond,
            )
            k_cpu = int(k.item()) if torch.is_tensor(k) else int(k)
            diffusion_output = noise_scheduler.step(
                model_output=noise_pred.detach().float().cpu(),
                timestep=k_cpu,
                sample=diffusion_output.detach().float().cpu(),
            ).prev_sample.to(device)

        actions_diffusion = get_action(diffusion_output, ACTION_STATS)
        return {"actions": actions_diffusion}

# Utils for Group Norm
def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
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
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
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