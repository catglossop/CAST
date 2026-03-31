
from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import pickle
import os
from PIL import Image
from etils import epath

RESIZE = (128, 128)
DATASET_ROOT_PATH = "/path/to/your/dataset_root"
ACTION_GEN_DIR = "/path/to/your/action_generation_outputs_dir"
TRAIN_VAL_SPLIT = 0.8

class CastCounterfactualDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Cast Counterfactual Dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(RESIZE[0], RESIZE[1], 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(3,),
                            dtype=np.float64,
                            doc='Robot state, consists of [2x position, 1x yaw]',
                        ),
                        'position': tfds.features.Tensor(
                            shape=(2,),
                            dtype=np.float64,
                            doc='Robot position',
                        ),
                        'yaw': tfds.features.Tensor(
                            shape=(1,),
                            dtype=np.float64,
                            doc='Robot yaw',
                        ),
                        'yaw_rotmat': tfds.features.Tensor(
                            shape=(3, 3),
                            dtype=np.float64,
                            doc='Robot yaw rotation matrix',
                        ),

                    }),
                    'action': tfds.features.Tensor(
                        shape=(2,),
                        dtype=np.float64,
                        doc='Robot action, consists of 2x position'
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float64,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float64,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                   'language_instruction': tfds.features.Tensor(
                        shape=(10,),
                        dtype=np.string_,
                        doc='Language Instruction.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                    'episode_id': tfds.features.Text(
                        doc='Episode ID.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path=DATASET_ROOT_PATH, split='train'),
            'val': self._generate_examples(path=DATASET_ROOT_PATH, split='val'),
        }

    def _generate_examples(self, path, split='train') -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""
        def _get_folder_names(data_dir):
            folder_names = [
                f for f in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, f))
                and "traj_data.pkl" in os.listdir(os.path.join(data_dir, f))
            ]   
            folder_names = [os.path.join(data_dir, f) for f in folder_names]

            return folder_names

        def _yaw_rotmat(yaw: float) -> np.ndarray:
            return np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ],)

        def _process_image(path, mode='stretch'):
            try:
                img = Image.open(path)
            except:
                img = Image.new('RGB', RESIZE, (255, 255, 255))
            if mode == 'stretch':
                img = img.resize(RESIZE)
            elif mode == 'crop':
                img = img.resize((85, 64))
                
                top = 0
                bottom = 64
                left = (85 - 64) // 2
                right = (85 + 64) // 2
                img = img.crop((left, top, right, bottom))
            
            return np.asarray(img, dtype='uint8')
        
        def _get_yaws_from_positions(positions: np.ndarray) -> np.ndarray:
            yaws = np.zeros(positions.shape[0])
            for i in range(1, positions.shape[0]):
                yaws[i] = np.arctan2(positions[i, 1] - positions[i-1, 1], positions[i, 0] - positions[i-1, 0])
            return yaws

        def _parse_example(episode_path, cf_data, idx):
            # load raw data --> this should change for your dataset
            data_path = os.path.join(episode_path, 'traj_data.pkl')
            data = np.load(data_path, allow_pickle=True)     # this is a list of dicts in our case
            if type(data["yaw"]) == list:
                data["yaw"] = np.array(data["yaw"])
            if data["position"].shape[0] < data["yaw"].shape[0]:
                data["yaw"] = data["yaw"][:data["position"].shape[0]]

            # Update with the counterfactual data at the end of the trajectory
            data["position"] = data["position"][:cf_data["start"]][cf_data["traj_start_idx"]:cf_data["traj_end_idx"]]
            data["position"] = np.concatenate([data["position"], cf_data["action"]], axis=0)
            computed_yaw = _get_yaws_from_positions(cf_data["action"])

            # Make sure the yaw is continuous with the previous yaw
            computed_yaw = computed_yaw - computed_yaw[0]
            computed_yaw = computed_yaw + data["yaw"][cf_data["start"]]
            
            # Now wrap the yaw to be between -pi and pi
            computed_yaw = (computed_yaw + np.pi) % (2 * np.pi) - np.pi
            data["yaw"] = data["yaw"][:cf_data["start"]+1][cf_data["traj_start_idx"]:cf_data["traj_end_idx"]]
            data["yaw"] = np.concatenate([data["yaw"], computed_yaw[1:]], axis=0)
  
            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i in range(len(data['position'])):

                # Get the language instruction
                language_instruction = cf_data["instruction"]
                language_instruction = [language_instruction] * 10
                    

                # Get image observation
                try: 
                    image_path = f'{i}.jpg'
                    img = _process_image(os.path.join(episode_path, image_path), mode='stretch')
                except:
                    img = Image.new('RGB', RESIZE, (255, 255, 255))

                # Get state observation
                position = data['position'][i]
                try:
                    yaw = data['yaw'][i].reshape(-1)
                except:
                    breakpoint()
                state = np.concatenate((position, yaw))
                yaw_rotmat = _yaw_rotmat(data["yaw"][i])
                episode.append({
                    'observation': {
                        'image': img,
                        'state': state,
                        'position': position,
                        'yaw': yaw,
                        'yaw_rotmat': yaw_rotmat,
                    },
                    'action': np.zeros(2), # We don't need the action, we base on the position later
                    'discount': 1.0,
                    'reward': float(i == (len(data['position']) - 1)),
                    'is_first': i == 0,
                    'is_last': i == (len(data['position']) - 1),
                    'is_terminal': i == (len(data) - 1),
                    'language_instruction': language_instruction,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path,
                    'episode_id': f"{cf_data['traj_path']}_{cf_data['start']}_{cf_data['atomic_command']}_{idx}",
                }
            }

            success = len(data['position']) >= 1

            if not success:
                return None
            # TFDS requires a *unique* string key per example (it is hashed for sharding).
            # Using only `episode_path` duplicates the key for every counterfactual row
            # under the same trajectory (same folder).
            example_key = sample["episode_metadata"]["episode_id"]
            return str(example_key), sample

        # Load the counterfactual data
        with open(os.path.join(ACTION_GEN_DIR, "action_generation_outputs.pkl"), "rb") as f:
            counterfactual_data = pickle.load(f)

        dataset_names = os.listdir(path)
        print(dataset_names)
        episode_paths = []
        for dataset_name in dataset_names:
            episode_paths += _get_folder_names(os.path.join(path, dataset_name))
    
        # split into train and val
        if split == 'train':
            episode_paths = episode_paths[:int(len(episode_paths) * TRAIN_VAL_SPLIT)]
        elif split == 'val':
            episode_paths = episode_paths[int(len(episode_paths) * TRAIN_VAL_SPLIT):]
        else:
            raise ValueError(f"Invalid split: {split}")

        # for smallish datasets, use single-thread parsing
        for episode_path in episode_paths:
            for idx, cf_data in enumerate(counterfactual_data[episode_path]):
                out = _parse_example(episode_path, cf_data, idx)
                if out is not None:
                    yield out


# When `tfds build` loads this file by path, `__module__` is `"__main__"`, so TFDS
# cannot infer a package directory for `dataset_info_from_configs` metadata and
# `joinpath()` is called with no segments (TypeError). Pin the config dir to this file's folder.
CastCounterfactualDataset.pkg_dir_path = epath.Path(__file__).parent
