import os
import argparse
import numpy as np
import yaml
import time
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from torch.optim import Adam, AdamW
from torchvision import transforms
import torch.backends.cudnn as cudnn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from warmup_scheduler import GradualWarmupScheduler
import wandb

from cast.atomic_model.model.atomic_model import AtomicModel
from cast.atomic_model.model.vision_encoder import VisionLanguageEncoder
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

from cast.atomic_model.data.dataset import AtomicDataset
from cast.atomic_model.train.training.train_utils import replace_bn_with_gn
from cast.atomic_model.train.training.train_eval_loop import train_eval_loop, load_model


def main(config):

    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if "gpu_ids" not in config:
            config["gpu_ids"] = [0]
        elif type(config["gpu_ids"]) == int:
            config["gpu_ids"] = [config["gpu_ids"]]
        print("Using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("Using cpu")

    first_gpu_id = config["gpu_ids"][0]
    device = torch.device(
        f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True

    cudnn.benchmark = True  # good if input sizes don't vary
    transform = ([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform = transforms.Compose(transform)

    # Load the data
    train_dataset = []
    train_dataset_lengths = []
    test_dataloaders = {}
    
    # Re-calibrate the weights on each of the datasets
    weights = []
    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]
        if "weight" not in data_config:
            data_config["weight"] = 1
        weights.append(data_config["weight"])
    total_weight = sum(weights)
    train_dataset_weights = [w / total_weight for w in weights]

    for dataset_idx, dataset_name in enumerate(config["datasets"]):
        data_config = config["datasets"][dataset_name]
        if "negative_mining" not in data_config:
            data_config["negative_mining"] = True
        if "goals_per_obs" not in data_config:
            data_config["goals_per_obs"] = 1
        if "end_slack" not in data_config:
            data_config["end_slack"] = 0
        if "waypoint_spacing" not in data_config:
            data_config["waypoint_spacing"] = 1

        for data_split_type in ["train", "test"]:
            if data_split_type in data_config:
                    dataset = AtomicDataset(
                        data_folder=data_config["data_folder"],
                        data_split_folder=data_config[data_split_type],
                        dataset_name=dataset_name,
                        image_size=config["image_size"],
                        waypoint_spacing=data_config["waypoint_spacing"],
                        len_traj_pred=config["len_traj_pred"],
                        context_size=config["context_size"],
                        end_slack=data_config["end_slack"],
                        normalize=True,
                        dataset_index=dataset_idx,
                    )
                    if data_split_type == "train":
                        if len(dataset) == 0:
                            print(f"Skipping dataset {dataset_name} for training")
                            print(dataset)
                            train_dataset_weights.pop(dataset_idx)
                            continue
                        else:
                            train_dataset.append(dataset)
                            train_dataset_lengths.append(len(dataset))
                    else:
                        dataset_type = f"{dataset_name}_{data_split_type}"
                        if dataset_type not in test_dataloaders:
                            test_dataloaders[dataset_type] = {}
                        test_dataloaders[dataset_type] = dataset

    # combine all the datasets from different robots
    train_dataset = ConcatDataset(train_dataset)
    data_dist = np.concatenate([[train_dataset_weights[i]/train_dataset_lengths[i]]*train_dataset_lengths[i] for i in range(len(train_dataset_lengths))])
    print("Length of train dataset", len(train_dataset))
    sampler = WeightedRandomSampler(data_dist, len(train_dataset), replacement=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        sampler = sampler,
        num_workers=config["num_workers"],
        drop_last=False,
        persistent_workers=config["num_workers"] > 0,
    )

    if "eval_batch_size" not in config:
        config["eval_batch_size"] = config["batch_size"]

    for dataset_type, dataset in test_dataloaders.items():
        if len(dataset) == 0:
            print(f"Skipping dataset {dataset_type} for evaluation")
            continue
        test_dataloaders[dataset_type] = DataLoader(
            dataset,
            batch_size=config["eval_batch_size"],
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

    vision_encoder = VisionLanguageEncoder(
        obs_encoder=config["obs_encoder"],
        obs_encoding_size=config["obs_encoding_size"],
        lang_encoding_size=config["lang_encoding_size"],
        context_size=config["context_size"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
    )
    vision_encoder = replace_bn_with_gn(vision_encoder)

    # Must match VisionLanguageEncoder.final_encoding_size (= obs_encoding_size // 4).
    # ConditionalUnet1D cats [diffusion_step_emb (256), global_cond], so cond_dim is 256 + this.
    global_cond_dim = config["obs_encoding_size"] // 4
    if config.get("encoding_size") != global_cond_dim:
        print(
            f"Note: encoding_size in config ({config.get('encoding_size')}) ignored for UNet; "
            f"using obs_encoding_size//4 = {global_cond_dim} to match the vision encoder output."
        )

    action_head = ConditionalUnet1D(
            input_dim=2,
            global_cond_dim=global_cond_dim,
            down_dims=config["down_dims"],
            cond_predict_scale=config["cond_predict_scale"],
        )

    noise_scheduler = DDPMScheduler(
            num_train_timesteps=config["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

    model = AtomicModel(
        vision_encoder=vision_encoder,
        action_head=action_head,
    )


    if config["clipping"]:
        print("Clipping gradients to", config["max_norm"])
        for p in model.parameters():
            if not p.requires_grad:
                continue
            p.register_hook(
                lambda grad: torch.clamp(
                    grad, -1 * config["max_norm"], config["max_norm"]
                )
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
        if config["scheduler"] == "cosine":
            print("Using cosine annealing with T_max", config["epochs"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["epochs"]*2, eta_min=1.0e-6
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
            if GradualWarmupScheduler is None:
                raise ImportError("warmup_scheduler is required when warmup=True")
            print("Using warmup scheduler")
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=config["warmup_epochs"],
                after_scheduler=scheduler,
            )

    current_epoch = 0
    if "load_run" in config:
        load_project_folder = os.path.join("logs", config["load_run"])
        print("Loading model from ", load_project_folder)
        latest_path = os.path.join(load_project_folder, "latest.pth")
        latest_checkpoint = torch.load(latest_path, map_location=device)
        load_model(model, latest_checkpoint)
        if "epoch" in latest_checkpoint:
            current_epoch = latest_checkpoint["epoch"] + 1

    # Multi-GPU
    if len(config["gpu_ids"]) > 1:
        model = nn.DataParallel(model, device_ids=config["gpu_ids"])
    model = model.to(device)

    if "load_run" in config:  # load optimizer and scheduler after data parallel
        if "optimizer" in latest_checkpoint:
            optimizer.load_state_dict(latest_checkpoint["optimizer"].state_dict())
        if scheduler is not None and "scheduler" in latest_checkpoint:
            scheduler.load_state_dict(latest_checkpoint["scheduler"].state_dict())
    

    train_eval_loop(
        train_model=config["train"],
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        noise_scheduler=noise_scheduler,
        train_loader=train_loader,
        test_dataloaders=test_dataloaders,
        transform=transform,
        epochs=config["epochs"],
        device=device,
        project_folder=config["project_folder"],
        print_log_freq=config["print_log_freq"],
        wandb_log_freq=config["wandb_log_freq"],
        image_log_freq=config["image_log_freq"],
        num_images_log=config["num_images_log"],
        current_epoch=current_epoch,
        use_wandb=config["use_wandb"],
        eval_fraction=config["eval_fraction"],
        eval_freq=config["eval_freq"],
        save_freq=config["save_freq"],
    ) 

    print("FINISHED TRAINING")


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Visual Navigation Transformer")

    # project setup
    parser.add_argument(
        "--config",
        "-c",
        default="configs/atomic_model.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )
    args = parser.parse_args()

    defaults_path = os.path.join(os.path.dirname(__file__), "configs", "defaults.yaml")
    if os.path.exists(defaults_path):
        with open(defaults_path, "r") as f:
            default_config = yaml.safe_load(f) or {}
    else:
        default_config = {}

    config = default_config

    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)

    config.update(user_config)

    config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    config["project_folder"] = os.path.join(
        "logs", config["project_name"], config["run_name"]
    )
    os.makedirs(
        config[
            "project_folder"
        ],  # should error if dir already exists to avoid overwriting and old project
    )

    if config.get("use_wandb", False):
        if wandb is None:
            raise ImportError("wandb is not installed but use_wandb=True in config")
        wandb.login()
        wandb.init(
            project=config["project_name"],
            settings=wandb.Settings(start_method="fork"),
        )
        # wandb defaults base_path to cwd for relative paths; resolved paths outside cwd
        # fail validation. Anchor base_path to the config file's directory.
        _cfg_path = os.path.abspath(args.config)
        wandb.save(_cfg_path, base_path=os.path.dirname(_cfg_path), policy="now")
        wandb.run.name = config["run_name"]
        # update the wandb args with the training configurations
        if wandb.run:
            wandb.config.update(config)
    all_gpu_ids = config.get("all_gpu_ids", config.get("gpu_ids", [0]))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in all_gpu_ids])
    print(config)
    main(config)
