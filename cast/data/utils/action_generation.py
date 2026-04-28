"""
Final CAST data-gen step: generate actions conditioned on atomic actions.

Outputs one (or more) `.pkl` files. Each `.pkl` contains a `List[Dict]` with:
  - `traj_path`
  - `instruction` (counterfactual_instruction)
  - `atomic_command` (counterfactual_action)
  - `action` (generated action / predicted waypoint chunk)
  - `action_idx` (index where the original atomic action started)
"""

from __future__ import annotations

import os
import glob
import pickle as pkl
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
from transformers import T5EncoderModel, T5Tokenizer

from cast.atomic_model.model.atomic_model import AtomicModel
from cast.atomic_model.model.vision_encoder import VisionLanguageEncoder
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from cast.diffusion_policy.diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from cast.atomic_model.train.training.train_utils import replace_bn_with_gn, get_action
from cast.atomic_model.train.training.train_eval_loop import load_model
from cast.atomic_model.utils.data_utils import resize_and_aspect_crop
from cast.atomic_model.train.training.train_utils import model_output

from cast.data.utils.common import base_actions, saved_batch_responses_path, get_trajectory_paths, load_trajectory_data, load_all_images_paths

def load_counterfactual_by_unique_id(config: dict) -> dict:
    path = saved_batch_responses_path(config, "counterfactual")
    by_id = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_id[row["unique_id"]] = row
    return by_id


def plot_trajectory_2d(
    positions: np.ndarray,
    *,
    action_chunk: np.ndarray,
    start: int,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    show: bool = False,
    save_path: Optional[str] = None,
    marker_start: str = "o",
    marker_end: str = "x",
    max_points: int = 50,
) -> None:
    """
    Plot a trajectory in 2D (x,y) plane for quick debugging.

    Args:
        positions: (N, 2) array.
    """
    if positions is None:
        return

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    orig_positions = np.asarray(positions)
    cf_positions = np.asarray(action_chunk)

    ax.plot(orig_positions[:, 0], orig_positions[:, 1], linewidth=2, alpha=0.9, label="Original")
    ax.plot(cf_positions[:, 0], cf_positions[:, 1], linewidth=2, alpha=0.9, label="Counterfactual")
    ax.scatter(orig_positions[start, 0], orig_positions[start, 1], marker=marker_start, s=60, label="Start")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    if title:
        ax.set_title(title)
    ax.legend()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    if fig is not None:
        plt.close(fig)


def _transform_images(pil_imgs: List[Image.Image], image_size: Tuple[int, int]) -> torch.Tensor:
    """Convert a list of PIL images into the model's expected tensor layout."""
    if not isinstance(pil_imgs, list):
        pil_imgs = [pil_imgs]

    transf_imgs = []
    for pil_img in pil_imgs:
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        # Keep preprocessing consistent with atomic_model training.
        transf_img = resize_and_aspect_crop(pil_img, image_resize_size=image_size)  # (3, H, W)
        transf_img = torch.unsqueeze(transf_img, 0)  # (1, 3, H, W)
        transf_imgs.append(transf_img)
    # Concatenate along channel dimension: (1, 3 * T, H, W)
    return torch.cat(transf_imgs, dim=1)


def t5_embed(text: str, *, device: torch.device) -> torch.Tensor:
    """Embed an atomic action string using T5 mean pooling."""
    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
    model = T5EncoderModel.from_pretrained("google-t5/t5-small").to(device)
    model.eval()

    tokens = tokenizer(text, return_tensors="pt", padding=True)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    text_features = model(tokens["input_ids"], attention_mask=tokens["attention_mask"]).last_hidden_state
    # mean pooling over tokens -> (1, hidden)
    text_features = text_features.mean(dim=1)
    return text_features


def get_model(model_config: Dict[str, Any]) -> Tuple[torch.nn.Module, DDPMScheduler]:
    """Build the atomic action model (vision encoder + action head)."""
    vision_encoder = VisionLanguageEncoder(
        obs_encoder=model_config["obs_encoder"],
        obs_encoding_size=model_config["obs_encoding_size"],
        lang_encoding_size=model_config["lang_encoding_size"],
        context_size=2,
        mha_num_attention_heads=model_config["mha_num_attention_heads"],
        mha_num_attention_layers=model_config["mha_num_attention_layers"],
        mha_ff_dim_factor=model_config["mha_ff_dim_factor"],
    )
    vision_encoder = replace_bn_with_gn(vision_encoder)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=model_config["num_diffusion_iters"],
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    action_head = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=model_config["encoding_size"],
        down_dims=model_config["down_dims"],
        cond_predict_scale=model_config["cond_predict_scale"],
    )

    model = AtomicModel(
        vision_encoder=vision_encoder,
        action_head=action_head,
    )
    return model, noise_scheduler

def build_cast_trajectory_records(
    config: dict
) -> list:
    """One record per trajectory: paths, summarize instructions, atomic segments."""

    counterfactual_by_id = load_counterfactual_by_unique_id(config)
    records = []
    for traj_path in get_trajectory_paths(config["dataset_path"]):
        uid_prefix = os.path.basename(traj_path.rstrip("/"))
        counterfactual_uids = [uid for uid in counterfactual_by_id if uid.startswith(uid_prefix)]
        for uid in counterfactual_uids:
            _, _, atomic_action_idx = int(uid.split("_")[-3]), int(uid.split("_")[-1]), int(uid.split("_aa_")[-1])
            counterfactual_action_instructions = counterfactual_by_id[uid].get("counterfactual_action_instruction", None)
            
            if counterfactual_action_instructions is None:
                continue
            records.append(
                {
                    "traj_path": traj_path,
                    "unique_id": uid,
                    "start": atomic_action_idx,
                    "counterfactual_action_instructions": counterfactual_action_instructions,
                    "atomic_action_idx": atomic_action_idx,
                }
            )
    return records


def generate_actions(config: Dict[str, Any]) -> None:
    """
    Load counterfactual generation output and create action-conditioned rollouts.

    Writes:
      {action_generation_output_dir}/{traj_uid}.pkl
      each file contains `List[Dict]`.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    action_gen_dir = os.path.join(config["output_dir"], "action_generation")
    os.makedirs(action_gen_dir, exist_ok=True)

    counterfactuals_path = saved_batch_responses_path(config, "counterfactual")
    if not os.path.exists(counterfactuals_path):
        raise FileNotFoundError(
            f"Counterfactual responses pkl not found at `{counterfactuals_path}`. "
            "Run the counterfactual step first (which should export `counterfactual_responses.jsonl`)."
        )

    # Get model config and load
    action_model_cfg = config.get("action_model", None)
    if action_model_cfg is None:
        raise KeyError("Missing `action_model` section in data_gen.yaml (needed to load the atomic model).")

    # atomic model checkpoint + config
    model_config_path = action_model_cfg["model_config"]
    if not os.path.exists(model_config_path):
        raise FileNotFoundError(f"Model config not found: `{model_config_path}`")
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)
    image_size = tuple(model_config.get("image_size", (96, 96)))

    ckpt_path = action_model_cfg["model_path"]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Atomic model checkpoint not found: `{ckpt_path}`")

    # Load data config
    data_config_path = config.get("data_config_path", None)
    if data_config_path is None:
        raise KeyError("Missing `data_config_path` in data_gen.yaml (needed to load the data config).")
    if not os.path.exists(data_config_path):
        raise FileNotFoundError(f"Data config not found: `{data_config_path}`")
    with open(data_config_path, "r") as f:
        data_config = yaml.safe_load(f)
    
    action_stats = data_config.get("action_stats", None)
    if action_stats is None:
        raise KeyError("Missing `action_stats` in data_config.yaml (needed to load the action stats).")

    # Build & load model.
    model, noise_scheduler = get_model(model_config)
    # The checkpoint may have been saved from CUDA; ensure it's loadable on CPU too.
    latest_checkpoint = torch.load(ckpt_path, map_location=device)
    load_model(model, latest_checkpoint)
    model.to(device)
    model.eval()

    # Precompute atomic action embeddings once.
    action_embeddings: Dict[str, torch.Tensor] = {}
    for action in base_actions:
        action_embeddings[action] = t5_embed(action, device=device)

    # Load counterfactual records
    traj_cache = {}
    records = build_cast_trajectory_records(config)

    viz_dir = config.get("viz_action_generation_dir", None)
    viz_max = int(config.get("viz_max_records", 0)) if viz_dir else 0
    viz_counter = 0

    outputs_by_traj = defaultdict(list)
    
    for rec in records:
        print(f"Processing record {rec['unique_id']}")
        traj_path = rec["traj_path"]
        dataset_name = traj_path.split("/")[-2]
        metric_waypoint_spacing = data_config[dataset_name].get("metric_waypoint_spacing", 0.12)
        if traj_path not in traj_cache:
            traj_data = load_trajectory_data(traj_path)
            traj_cache[traj_path] = traj_data
        else:
            traj_data = traj_cache[traj_path]

        start = rec["start"]

        # Get the actual start and end indices of the trajectory
        traj_start_idx = rec["unique_id"].split("_")[-5]
        traj_end_idx = rec["unique_id"].split("_")[-3]
        if not int(traj_start_idx) <= start <= int(traj_end_idx):
            continue

        counterfactual_action_instructions = rec["counterfactual_action_instructions"]

        # Prepare batched inputs for this trajectory only
        batch_size = model_config.get("batch_size", 1)
        # Only batch over this trajectory's instructions
        instrs = []
        cmds = []
        context_tensors = []
        last_positions = []
        last_yaws = []
        valid_idxs = []  # to keep which instructions in the batch are valid

        img_paths_all = load_all_images_paths(traj_path)

        for idx, counterfactual_action_instruction in enumerate(counterfactual_action_instructions):
            instruction = counterfactual_action_instruction.get("counterfactual_instruction", None)
            command = counterfactual_action_instruction.get("counterfactual_action", None)
            if instruction is None or command is None:
                continue
            min_context_idx = max(0, start - model_config["context_size"])
            img_paths = img_paths_all[min_context_idx:start + 1]
            context_imgs = [Image.open(img_path) for img_path in img_paths]
            if len(context_imgs) < model_config["context_size"]:
                print(f"Not enough images for context size {model_config['context_size']} at {traj_path} {start}")
                continue
            context_tensor = _transform_images(context_imgs, image_size).to(device)
            instrs.append(instruction)
            cmds.append(command)
            context_tensors.append(context_tensor)
            last_positions.append(np.asarray(traj_data["position"][start], dtype=np.float32))  # (2,)
            try:
                last_yaws.append(float(traj_data["yaw"][start]))
            except:
                last_yaws.append(float(traj_data["yaw"][start][0]))
            valid_idxs.append(idx)

        # Now process in batches
        total = len(instrs)
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_contexts = torch.cat(context_tensors[batch_start:batch_end], dim=0)
            batch_cmds = cmds[batch_start:batch_end]
            batch_instrs = instrs[batch_start:batch_end]
            batch_last_positions = last_positions[batch_start:batch_end]
            batch_last_yaws = last_yaws[batch_start:batch_end]

            # embed each atomic command and stack
            batch_prompt_embeddings = torch.cat(
                [action_embeddings[cmd].to(device).unsqueeze(0) for cmd in batch_cmds], dim=0
            )  # (B, D)
            # Some models expect embeddings to be (B, 1, D)
            batch_prompt_embeddings = batch_prompt_embeddings

            rollout = model_output(
                model=model,
                noise_scheduler=noise_scheduler,
                batch_obs_images=batch_contexts,
                batch_lang_embeddings=batch_prompt_embeddings,
                pred_horizon=model_config["len_traj_pred"],
                action_dim=2,
                num_samples=1,
                batch_size=batch_end - batch_start,
                device=device,
            )  # (B, T, 2) or similar

            actions_batched = rollout["actions"].detach().cpu().numpy()  # (B, T, 2)
            # Match propose_counterfactuals_gemini.py's normalization/coordinate transforms.
            actions_batched = actions_batched - actions_batched[:, [0], :]

            for sample_idx in range(actions_batched.shape[0]):
                rollout_sample = actions_batched[sample_idx:sample_idx+1]
                last_pos = batch_last_positions[sample_idx]  # (2,)
                last_yaw = batch_last_yaws[sample_idx]
                rot_mat = np.array(
                    [[np.cos(last_yaw), -np.sin(last_yaw)], [np.sin(last_yaw), np.cos(last_yaw)]],
                    dtype=np.float32,
                ).reshape(-1, 2, 2)  # (1, 2, 2)
                rollout_rotated = rot_mat @ np.transpose(rollout_sample, (0, 2, 1))  # (1, 2, T)
                rollout_rotated = np.transpose(rollout_rotated, (0, 2, 1))  # (1, T, 2)
                rollout_rotated = rollout_rotated * float(metric_waypoint_spacing)
                rollout_abs = rollout_rotated + np.expand_dims(last_pos, 0)

                # Store only the first sample (always num_samples=1)
                action_chunk = rollout_abs[0]
                outputs_by_traj[traj_path].append(
                    {
                        "traj_path": traj_path,
                        "instruction": batch_instrs[sample_idx],
                        "atomic_command": batch_cmds[sample_idx],
                        "start": start,
                        "action": action_chunk,
                        "traj_start_idx": int(traj_start_idx),
                        "traj_end_idx": int(traj_end_idx),
                    }
                )

    with open(os.path.join(action_gen_dir, "action_generation_outputs.pkl"), "wb") as f:
        pkl.dump(outputs_by_traj, f)
