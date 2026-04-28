import json
import os

import numpy as np
from google.cloud import storage

from cast.data.utils.common import (
    convert_path_to_uri,
    gcs_response_path_txt_for_job,
    get_trajectory_paths,
    load_all_images_paths,
    make_request,
    process_job,
    process_responses,
    saved_batch_responses_path,
    upload_batches_to_bucket,
    formats,
)


def load_describe_rows(config: dict) -> list:
    path = saved_batch_responses_path(config, "hindsight_describe")
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def ordered_descriptions_for_trajectory(describe_rows: list, traj_path: str) -> list:
    """Match hindsight_describe outputs to a trajectory using the `path` field."""
    traj_norm = os.path.normpath(traj_path)
    matched = []
    for r in describe_rows:
        p = r.get("path") or ""
        if not p:
            continue
        if os.path.normpath(os.path.dirname(p)) != traj_norm:
            continue
        matched.append(r)
    matched.sort(
        key=lambda r: int(os.path.splitext(os.path.basename(r["path"]))[0])
    )
    return [(r["image_description"], r["path"]) for r in matched]


def hindsight_describe(config: dict, schema: dict, prompt: str) -> None:
    """Pairwise image descriptions (VLM step of the hierarchical pipeline)."""
    job_name = "hindsight_describe"

    os.makedirs(os.path.dirname(gcs_response_path_txt_for_job(config, job_name)), exist_ok=True)

    # os.makedirs(config["hindsight_output_dir"] + "/" + job_name, exist_ok=True)
    client = storage.Client(project=config["gcp_project_id"])

    traj_paths = get_trajectory_paths(config["dataset_path"])
    bucket = client.bucket(config["base_uri"])
    requests = []

    for traj_path in traj_paths:
        image_paths = load_all_images_paths(traj_path)[:: config["image_sampling_rate"]]
        for img_idx in range(1, len(image_paths)):
            current_img_path = convert_path_to_uri(config, image_paths[img_idx])
            previous_img_path = convert_path_to_uri(config, image_paths[img_idx - 1])
            modified_prompt = prompt.format(path=image_paths[img_idx])
            request = make_request(
                modified_prompt, [current_img_path, previous_img_path], schema
            )
            requests.append(request)

    upload_batches_to_bucket(config, requests, bucket, job_name)

    gcs_path_file = gcs_response_path_txt_for_job(config, job_name)
    os.makedirs(os.path.dirname(gcs_path_file), exist_ok=True)
    if not os.path.exists(gcs_path_file):
        print("*" * 100)
        print("Processing job for hindsight describe. This may take a while...")
        response_path = process_job(config, job_name)
        with open(gcs_path_file, "w") as f:
            f.write(response_path)
    else:
        print("*" * 100)
        print("Job for hindsight describe already processed. Loading response path from file...")
        print(
            "If you'd like to reprocess the job, delete the gcs_response_path.txt file and run the function again."
        )
        with open(gcs_path_file, "r") as f:
            response_path = f.read().strip("\n")

    process_responses(config, response_path, job_name)


def hindsight_summarize(config: dict, prompt: str, schema: dict) -> None:
    """
    LLM step: aggregate pairwise descriptions into trajectory-level instructions.

    Uses the hierarchical prompts (init + instruct_prompt_llm) from
    `prompt_gemini_hierarchical.json`, matching the relabelling script, while
    keeping the same Vertex batch request layout as other CAST steps.
    """
    job_name = "hindsight_summarize"
    os.makedirs(os.path.dirname(gcs_response_path_txt_for_job(config, job_name)), exist_ok=True)

    client = storage.Client(project=config["gcp_project_id"])

    describe_path = saved_batch_responses_path(config, "hindsight_describe")
    if not os.path.exists(describe_path):
        raise ValueError(
            "Hindsight describe responses not found. Run the hindsight describe step first."
        )

    describe_rows = load_describe_rows(config)

    traj_paths = get_trajectory_paths(config["dataset_path"])
    bucket = client.bucket(config["base_uri"])
    requests = []

    for traj_path in traj_paths:
        descriptions = ordered_descriptions_for_trajectory(describe_rows, traj_path)
        if not descriptions:
            print(
                f"Skipping hindsight_summarize for {traj_path}: no describe lines matched."
            )
            continue
        total_traj_len = len(descriptions)
        i = 0
        while i < total_traj_len:
            traj_len = np.random.choice(np.arange(config["min_traj_len"], config["max_traj_len"] + 1))
            start = i
            end = min(i + traj_len, total_traj_len)

            descriptions_chunk = descriptions[start:end]
            joined_descriptions = ""
            for d, _ in descriptions_chunk:
                joined_descriptions += f"{i + 1}. {d}\n"
                i += 1
            start_in_traj = int(os.path.splitext(os.path.basename(descriptions_chunk[0][1]))[0])
            end_in_traj = int(os.path.splitext(os.path.basename(descriptions_chunk[-1][1]))[0])
            unique_id = f"{os.path.basename(traj_path.rstrip('/'))}_start_{start_in_traj}_end_{end_in_traj}"
            modified_prompt = prompt.format(instructions=joined_descriptions, unique_id=unique_id, formats=formats)
            request = make_request(modified_prompt, [], schema)
            requests.append(request)
            i += traj_len
    if not requests:
        raise ValueError("No hindsight summarize requests were built (empty dataset?).")

    upload_batches_to_bucket(config, requests, bucket, job_name)

    gcs_path_file = gcs_response_path_txt_for_job(config, job_name)
    os.makedirs(os.path.dirname(gcs_path_file), exist_ok=True)
    if not os.path.exists(gcs_path_file):
        print("*" * 100)
        print("Processing job for hindsight summarize. This may take a while...")
        response_path = process_job(config, job_name)
        with open(gcs_path_file, "w") as f:
            f.write(response_path)
    else:
        print("*" * 100)
        print("Job for hindsight summarize already processed. Loading response path from file...")
        print(
            "If you'd like to reprocess the job, delete the gcs_response_path.txt file and run the function again."
        )
        with open(gcs_path_file, "r") as f:
            response_path = f.read().strip("\n")

    process_responses(config, response_path, job_name)
