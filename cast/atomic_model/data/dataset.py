import numpy as np
import os
import pickle
import yaml
from typing import Any, Dict, List, Optional, Tuple
import tqdm
import io
import lmdb
import glob

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from transformers import T5EncoderModel, T5Tokenizer

from cast.atomic_model.utils.data_utils import (
    img_path_to_data,
    calculate_sin_cos,
    get_data_path,
    to_local_coords,
    get_image_paths,
)

class AtomicDataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        waypoint_spacing: int,
        len_traj_pred: int,
        context_size: int,
        end_slack: int = 0,
        normalize: bool = True,
        atomic_action_commands: List[str] = ["turn right", "turn left", "go forward", "stop", "adjust left", "adjust right"],
        dataset_index: int = 0,
    ):
        """
        Main atomicdataset class

        Args:
            data_folder (string): Directory with all the image data
            data_split_folder (string): Directory with filepaths.txt, a list of all trajectory names in the dataset split that are each seperated by a newline
            dataset_name (string): Name of the dataset [recon, go_stanford, scand, tartandrive, etc.]
            waypoint_spacing (int): Spacing between waypoints
            len_traj_pred (int): Length of trajectory of waypoints to predict if this is an action dataset
            context_size (int): Number of previous observations to use as context
            end_slack (int): Number of timesteps to ignore at the end of the trajectory
            normalize (bool): Whether to normalize the distances or actions
        """
        self.data_folder = data_folder
        self.data_split_folder = data_split_folder
        self.dataset_name = dataset_name
        
        traj_names_file = os.path.join(data_split_folder, "traj_names.txt")
        with open(traj_names_file, "r") as f:
            file_lines = f.read()
            self.traj_names = file_lines.split("\n")
        if "" in self.traj_names:
            self.traj_names.remove("")

        self.image_size = image_size
        self.waypoint_spacing = waypoint_spacing

        self.len_traj_pred = len_traj_pred

        self.context_size = context_size
        self.end_slack = end_slack
        self.normalize = normalize
        self.atomic_action_commands = atomic_action_commands
        self.dataset_index = dataset_index

        self.trajectory_cache = {}
        self._load_index()
        self._build_caches()
        self._build_language_embeddings()
        

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_image_cache"] = None
        return state
    
    def __setstate__(self, state):
        self.__dict__ = state
        self._build_caches()
    
    def _build_language_embeddings(self):
        self.language_embeddings = {}
        language_tokenizer = T5Tokenizer.from_pretrained("t5-small")
        language_model = T5EncoderModel.from_pretrained("t5-small")
        language_model.eval()
        with torch.no_grad():
            for command in self.atomic_action_commands:
                inputs = language_tokenizer(command, return_tensors="pt")
                outputs = language_model(**inputs)
                # Detach: non-leaf tensors with requires_grad cannot be sent across DataLoader workers.
                emb = outputs.last_hidden_state.mean(dim=1).detach().cpu().clone()
                self.language_embeddings[command] = emb

    def _build_caches(self, use_tqdm: bool = True):
        """
        Build a cache of images for faster loading using LMDB
        """
        cache_filename = os.path.join(
            self.data_split_folder,
            f"dataset_{self.dataset_name}.lmdb",
        )

        # Load all the trajectories into memory. These should already be loaded, but just in case.
        for traj_name in self.traj_names:
            self._get_trajectory(traj_name)

        """
        If the cache file doesn't exist, create it by iterating through the dataset and writing each image to the cache
        """
        if not os.path.exists(cache_filename):
            tqdm_iterator = tqdm.tqdm(
                self.traj_names,
                disable=not use_tqdm,
                dynamic_ncols=True,
                desc=f"Building LMDB cache for {self.dataset_name}"
            )
            with lmdb.open(cache_filename, map_size=2**40) as image_cache:
                with image_cache.begin(write=True) as txn:
                    for traj_name in tqdm_iterator:
                        image_paths = get_image_paths(self.data_folder, traj_name)
                        for image_path in image_paths:
                            with open(image_path, "rb") as f:
                                txn.put(image_path.encode(), f.read())

        # Reopen the cache file in read-only mode
        self._image_cache: lmdb.Environment = lmdb.open(cache_filename, readonly=True)

    def _build_index(self, use_tqdm: bool = False):
        """
        Build an index consisting of tuples (trajectory name, time)
        """
        samples_index = []

        for traj_name in tqdm.tqdm(self.traj_names, disable=not use_tqdm, dynamic_ncols=True):
            traj_data = self._get_trajectory(traj_name)
            traj_len = len(traj_data["position"])
            begin_time = self.context_size * self.waypoint_spacing
            end_time = traj_len - self.end_slack - self.len_traj_pred * self.waypoint_spacing
            for curr_time in range(begin_time, end_time):
                samples_index.append((traj_name, curr_time))
        return samples_index

    def _load_index(self) -> None:
        """
        Generates a list of tuples of (obs_traj_name, obs_time) for each observation in the dataset
        """
        index_to_data_path = os.path.join(
            self.data_split_folder,
            f"dataset_context_{self.context_size}_slack_{self.end_slack}.pkl",
        )
        try:
            # load the index_to_data if it already exists (to save time)
            with open(index_to_data_path, "rb") as f:
                self.index_to_data = pickle.load(f)
        except:
            # if the index_to_data file doesn't exist, create it
            self.index_to_data= self._build_index()
            with open(index_to_data_path, "wb") as f:
                pickle.dump((self.index_to_data), f)

    def _load_image(self, trajectory_name, time):
        image_path = get_data_path(self.data_folder, trajectory_name, time)
        try:
            with self._image_cache.begin() as txn:
                image_buffer = txn.get(image_path.encode())
                image_bytes = bytes(image_buffer)
            image_bytes = io.BytesIO(image_bytes)
            return img_path_to_data(image_bytes, self.image_size)
        except TypeError:
            print(f"Failed to load image {image_path}")

    def _compute_actions(self, traj_data, curr_time):
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred * self.waypoint_spacing + 1
        yaw = traj_data["yaw"][start_index:end_index:self.waypoint_spacing]
        positions = traj_data["position"][start_index:end_index:self.waypoint_spacing]

        if type(yaw) == list:
            yaw = np.array(yaw)

        if len(yaw.shape) == 2:
            yaw = yaw.squeeze(1)

        if yaw.shape != (self.len_traj_pred + 1,):
            const_len = self.len_traj_pred + 1 - yaw.shape[0]
            yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
            positions = np.concatenate([positions, np.repeat(positions[-1][None], const_len, axis=0)], axis=0)
        yaw = yaw[:self.len_traj_pred + 1]
        positions = positions[:self.len_traj_pred + 1, :]
        assert yaw.shape == (self.len_traj_pred + 1,), f"{yaw.shape} and {(self.len_traj_pred + 1,)} should be equal"
        assert positions.shape == (self.len_traj_pred + 1, 2), f"{positions.shape} and {(self.len_traj_pred + 1, 2)} should be equal"

        waypoints = to_local_coords(positions, positions[0], yaw[0])

        assert waypoints.shape == (self.len_traj_pred + 1, 2), f"{waypoints.shape} and {(self.len_traj_pred + 1, 2)} should be equal"

        actions = waypoints[1:]
        
        if self.normalize:
            actions[:, :2] /= traj_data["metric_waypoint_spacing"] * self.waypoint_spacing

        assert actions.shape == (self.len_traj_pred, 2), f"{actions.shape} and {(self.len_traj_pred, 2)} should be equal"

        return actions, traj_data["metric_waypoint_spacing"]
    
    def _get_trajectory(self, trajectory_name):
        if trajectory_name in self.trajectory_cache:
            return self.trajectory_cache[trajectory_name]
        else:
            with open(os.path.join(self.data_folder, trajectory_name, f"traj_data.pkl"), "rb") as f:
                traj_data = pickle.load(f)
            self.trajectory_cache[trajectory_name] = traj_data
            return traj_data

    def __len__(self) -> int:
        return len(self.index_to_data)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        """
        Args:
            i (int): index to ith datapoint
        Returns:
            Tuple of tensors containing the context, observation, action label
                obs_image (torch.Tensor): tensor of shape [3, H, W] containing the image of the robot's observation
                action_label (torch.Tensor): tensor of shape (T, 2) containing the action labels from the observation to the goal
                which_dataset (torch.Tensor): index of the datapoint in the dataset [for identifying the dataset for visualization when using multiple datasets]
        """
        f_curr, curr_time = self.index_to_data[i]

        # Load images
        context = []

        # Sample the last self.context_size times from interval [0, curr_time)
        context_times = list(
            range(
                curr_time + -self.context_size * self.waypoint_spacing,
                curr_time + 1,
                self.waypoint_spacing,
            )
        )
        context = [(f_curr, t) for t in context_times]


        try:
            obs_image = torch.cat([
                self._load_image(f, t) for f, t in context
            ])
        except:
            print(context)
            breakpoint()

        # Load other trajectory data
        curr_traj_data = self._get_trajectory(f_curr)
        curr_traj_len = len(curr_traj_data["position"])
        assert curr_time < curr_traj_len, f"{curr_time} and {curr_traj_len}"

        # Load language embeddings
        lang = curr_traj_data["language_instruction"]
        lang_embed = self.language_embeddings[lang]

        # Compute actions
        actions, waypoint_spacing = self._compute_actions(curr_traj_data, curr_time)
        actions_torch = torch.as_tensor(actions, dtype=torch.float32)
        
        return (
            torch.as_tensor(obs_image, dtype=torch.float32),
            actions_torch,
            torch.as_tensor(lang_embed, dtype=torch.float32),
            lang,
            self.dataset_name,
            torch.as_tensor(waypoint_spacing, dtype=torch.float32),
        )
