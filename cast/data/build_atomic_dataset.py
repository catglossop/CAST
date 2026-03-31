import tyro
import yaml
import pickle as pkl
import os
import shutil
from tqdm import tqdm
from cast.data.utils.atomic_decomposition import discretize_trajectory
from cast.data.utils.common import get_trajectory_paths, load_trajectory_data, base_actions

def build_atomic_dataset(config: dict, rebuild: bool = False) -> None:

    with open(config["data_config_path"], "r") as f:
        data_config = yaml.safe_load(f)
    
    datasets = config["datasets"]

    # Specific to GNM datasets, we store the metric spacing of the waypoints in data_config.yaml
    metric_waypoint_spacing = { dataset : data_config[dataset]["metric_waypoint_spacing"] for dataset in datasets}
    
    if rebuild: 
        shutil.rmtree(f"{config['output_dir']}/atomic_dataset")

    for base_action in base_actions: 
        atomic_output_dir = f"{config['output_dir']}/atomic_dataset/{('_').join(base_action.split(' '))}"
        os.makedirs(atomic_output_dir, exist_ok=True)

    for traj_path in tqdm(get_trajectory_paths(config["dataset_path"])):
        dataset_name = traj_path.split("/")[-2]
        traj_data = load_trajectory_data(traj_path)
        segments = discretize_trajectory(traj_path, config)
        for segment in segments:
            segment_traj_data = {key: traj_data[key][segment["start"]:segment["end"]] for key in traj_data.keys()}
            segment_traj_data["metric_waypoint_spacing"] = metric_waypoint_spacing[dataset_name]
            segment_traj_data["language_instruction"] = segment["label"]
            segment_traj_data["dataset_name"] = dataset_name
            atomic_output_dir = f"{config['output_dir']}/atomic_dataset/{('_').join(segment['label'].split(' '))}"
            traj_name = traj_path.split("/")[-1]
            traj_output_dir = f"{atomic_output_dir}/{traj_name}_{segment['start']}_{segment['end']}"
            os.makedirs(traj_output_dir, exist_ok=True)
            with open(f"{traj_output_dir}/traj_data.pkl", "wb") as f:
                pkl.dump(segment_traj_data, f)
            image_paths = [f"{traj_path}/{i}.jpg" for i in range(segment['start'], segment['end']+1)]
            with open(f"{traj_output_dir}/image_paths.txt", "w") as f:
                for image_path in image_paths:
                    f.write(f"{image_path}\n")

        

def main(config_path: str,
         rebuild: bool = False,
        ) -> None:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    build_atomic_dataset(config, rebuild)

if __name__ == "__main__":
    tyro.cli(main)