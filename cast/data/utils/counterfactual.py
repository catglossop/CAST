"""Counterfactual proposals with Vertex batch (same flow as hindsight_describe)."""
import json
import os
import pickle as pkl

from google.cloud import storage

from cast.data.utils.atomic_decomposition import discretize_trajectory
from cast.data.utils.common import (
    convert_path_to_uri,
    gcs_response_path_txt_for_job,
    make_request,
    process_job,
    process_responses,
    saved_batch_responses_path,
    upload_batches_to_bucket,
    base_actions,
    get_trajectory_paths,
    join_string_list,
)


def export_cf_responses_pickle(config: dict) -> list:
    """Load parsed CF outputs from local jsonl (after process_responses) and save pickle."""
    path = saved_batch_responses_path(config, "cf")
    saved = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            saved.append(json.loads(line))

    local_save_path = config.get("local_save_path", config["output_dir"])
    os.makedirs(local_save_path, exist_ok=True)
    out_path = os.path.join(local_save_path, "cast_cf_responses.pkl")
    with open(out_path, "wb") as f:
        pkl.dump(saved, f)
    return saved

def load_hindsight_by_unique_id(config: dict) -> dict:
    path = saved_batch_responses_path(config, "filter")
    by_id = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_id[row["unique_id"]] = row
    return by_id

def build_cast_trajectory_records(
    config: dict
) -> list:
    """One record per trajectory: paths, summarize instructions, atomic segments."""

    hindsight_by_id = load_hindsight_by_unique_id(config)
    records = []
    for traj_path in get_trajectory_paths(config["dataset_path"]):
        uid_prefix = os.path.basename(traj_path.rstrip("/"))
        hindsight_uids = [uid for uid in hindsight_by_id if uid.startswith(uid_prefix)]
        segments = discretize_trajectory(traj_path, config)
        for uid in hindsight_uids:
            start, end = int(uid.split("_")[-3]), int(uid.split("_")[-1])
            instructions = hindsight_by_id[uid].get("best", []) + hindsight_by_id[uid].get("new", [])
            chunk_segments = [s for s in segments if (s["start"] >= start and s["end"] <= end) or 
                                                        (s["start"] <= start and start <= s["end"] <= end) or 
                                                        (s["start"] <= end and end <= s["end"])]

            records.append(
                {
                    "traj_path": traj_path,
                    "unique_id": uid,
                    "hindsight_instructions": instructions,
                    "atomic_segments": chunk_segments,
                }
            )
    return records


def counterfactual_propose(config: dict, schema: dict, prompt: str) -> None:
    """
    Build CF batch requests, run Vertex job, process_responses, export to pickle format for conversion

    """
    job_name = "counterfactual"

    os.makedirs(os.path.dirname(gcs_response_path_txt_for_job(config, job_name)), exist_ok=True)

    records = build_cast_trajectory_records(config)

    client = storage.Client(project=config["gcp_project_id"])
    bucket = client.bucket(config["base_uri"])
    requests = []

    for rec in records:
        traj_path = rec["traj_path"]
        uid_base = rec["unique_id"]
        start, end = int(uid_base.split("_")[-3]), int(uid_base.split("_")[-1])
        hindsight_instructions = rec["hindsight_instructions"]
        segments = rec["atomic_segments"]

        if not segments:
            continue

        for seg_idx in range(len(segments)):
            if not start <= segments[seg_idx]["start"] <= end:
                continue
            cum_labels = [s["label"] for s in segments[:seg_idx]]

            labels_str = join_string_list(cum_labels)
            hindsight_instructions_str = join_string_list(hindsight_instructions)
            base_actions_str = join_string_list(base_actions)

            curr_atomic_action = segments[seg_idx]["label"]

            image_path = os.path.join(traj_path, f"{segments[seg_idx]['start']}.jpg")
            image_uri = convert_path_to_uri(config, image_path)
            cf_uid = f"{uid_base}_aa_{segments[seg_idx]['start']}"

            modified_prompt = prompt.format(labels=labels_str, 
                                            curr_atomic_action=curr_atomic_action, 
                                            base_actions=base_actions_str, 
                                            instructions=hindsight_instructions_str, 
                                            unique_id=cf_uid)
            requests.append(make_request(modified_prompt, [image_uri], schema))

    if not requests:
        raise ValueError(
            "No counterfactual requests were built (empty atomic segments or missing frames)."
        )

    upload_batches_to_bucket(config, requests, bucket, job_name)

    gcs_path_file = gcs_response_path_txt_for_job(config, job_name)
    if config.get("run_cf_batch_job", True):
        if not os.path.exists(gcs_path_file):
            print("*" * 100)
            print("Processing job for counterfactuals. This may take a while...")
            response_path = process_job(config, job_name)
            with open(gcs_path_file, "w") as f:
                f.write(response_path)
        else:
            print("*" * 100)
            print("Job for counterfactuals already processed. Loading response path from file...")
            print(
                "If you'd like to reprocess the job, delete the gcs_response_path.txt file and run again."
            )
            with open(gcs_path_file, "r") as f:
                response_path = f.read().strip("\n")
    else:
        with open(gcs_path_file, "r") as f:
            response_path = f.read().strip("\n")

    process_responses(config, response_path, job_name)
