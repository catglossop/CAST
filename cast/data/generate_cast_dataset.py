
import os
import json
import yaml
from tqdm import tqdm
import tyro

# from cast.data.utils.atomic_decomposition import 
from cast.data.utils.hindsight_labeling import hindsight_describe, hindsight_summarize
from cast.data.utils.common import setup_gcp
from cast.data.utils.filtering import filtering
from cast.data.utils.counterfactual import counterfactual_propose
from cast.data.utils.action_generation import generate_actions

def hindsight_labeling_step(config: dict) -> None:

    with open(config["hindsight_describe_schema"], "r") as f:
        hindsight_describe_schema = json.load(f)
    with open(config["hindsight_summarize_schema"], "r") as f:
        hindsight_summarize_schema = json.load(f)
    with open(config["hindsight_describe_prompt"], "r") as f:
        hindsight_describe_prompt = f.read()
    with open(config["hindsight_summarize_prompt"], "r") as f:
        hindsight_summarize_prompt = f.read()

    os.makedirs(config["hindsight_output_dir"], exist_ok=True)

    hindsight_describe(config, hindsight_describe_schema, hindsight_describe_prompt)
    hindsight_summarize(config, hindsight_summarize_prompt, hindsight_summarize_schema)


def filtering_step(config: dict) -> None:

    with open(config["filter_step_schema"], "r") as f:
        schema = json.load(f)
    with open(config["filter_step_prompt"], "r") as f:
        prompt = f.read()

    filtering(config, schema, prompt)


def counterfactual_generation_step(config: dict) -> None:

    with open(config["cf_step_schema"], "r") as f:
        schema = json.load(f)
    with open(config["cf_step_prompt"], "r") as f:
        prompt = f.read()

    counterfactual_propose(config, schema, prompt)

def action_generation_step(config: dict) -> None:
    generate_actions(config)

def main(
    config_path: str = "config.yaml",
    hindsight_step: bool = False,
    filter_step: bool = False,
    cf_step: bool = False,
    action_gen_step: bool = False,
    run_all: bool = False,
    force_dataset_reupload: bool = False,
    incremental_dataset_upload: bool = False,
) -> None:
    """
    Data is assumed to be in form of:
     - traj_directory/
        - traj_data.pkl
            - "position": [x, y]
            - "yaw" : [yaw]
        - 0.jpg
        - 1.jpg
        - ...
        - n.jpg
    Args:
        config_path: Path to the config file (prompts, schemas, paths, model id, thresholds).
        atomic_step: Run odometry-based atomic decomposition.
        hindsight_step: Run pairwise describe + hierarchical summarize (Vertex batch).
        filter_step: Run filter batch (see utils.filtering.filtering).
        cf_step: Run counterfactual batch (see utils.counterfactual.counterfactual_propose).
    """

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(config["output_dir"], exist_ok=True)

    setup_gcp(config, force_dataset_reupload, incremental_dataset_upload)

    if run_all or hindsight_step:
        hindsight_labeling_step(config)

    if run_all or filter_step:
        filtering_step(config)

    if run_all or cf_step:
        counterfactual_generation_step(config)
    
    if run_all or action_gen_step: 
        action_generation_step(config)


if __name__ == "__main__":
    tyro.cli(main)
