"""Filter trajectories with Vertex batch (same flow as hindsight_describe)."""
import json
import os
import pickle as pkl
from typing import Optional

from google.cloud import storage

from cast.data.utils.atomic_decomposition import discretize_trajectory
from cast.data.utils.common import (
    gcs_response_path_txt_for_job,
    get_trajectory_paths,
    make_request,
    process_job,
    process_responses,
    saved_batch_responses_path,
    upload_batches_to_bucket,
    join_string_list,
)


def load_summarize_by_unique_id(config: dict) -> dict:
    path = saved_batch_responses_path(config, "hindsight_summarize")
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
    config: dict, summarize_by_id: Optional[dict] = None
) -> list:
    """One record per trajectory: paths, summarize instructions, atomic segments."""
    if summarize_by_id is None:
        summarize_by_id = load_summarize_by_unique_id(config)
    records = []
    for traj_path in get_trajectory_paths(config["dataset_path"]):
        uid_prefix = os.path.basename(traj_path.rstrip("/"))
        summary_uids = [uid for uid in summarize_by_id if uid.startswith(uid_prefix)]
        segments = discretize_trajectory(traj_path, config)
        for uid in summary_uids:
            start, end = int(uid.split("_")[-3]), int(uid.split("_")[-1])
            summ = summarize_by_id[uid]
            instructions = summ.get("instructions") or []
            chunk_segments = [s["label"] for s in segments if (s["start"] >= start and s["end"] <= end) or 
                                                        (s["start"] <= start and start <= s["end"] <= end) or 
                                                        (s["start"] <= end and end <= s["end"])]

            records.append(
                {
                    "traj_path": traj_path,
                    "unique_id": uid,
                    "orig_instructions": instructions,
                    "atomic_segments": chunk_segments,
                }
            )
    return records

def filtering(config: dict, schema: dict, prompt: str) -> None:
    """
    Build filter batch requests, run Vertex job, run process_responses, merge + pickle.
    """
    job_name = "filter"

    os.makedirs(os.path.dirname(gcs_response_path_txt_for_job(config, job_name)), exist_ok=True)

    summarize_by_id = load_summarize_by_unique_id(config)
    records = build_cast_trajectory_records(config, summarize_by_id=summarize_by_id)

    client = storage.Client(project=config["gcp_project_id"])
    bucket = client.bucket(config["base_uri"])
    requests = []

    for record in records:

        orig_instructions = join_string_list(record["orig_instructions"])
        labels = join_string_list(record["atomic_segments"])
        unique_id = record["unique_id"]

        modified_prompt = prompt.format(orig_instructions=orig_instructions, labels=labels, unique_id=unique_id)
        requests.append(make_request(prompt=modified_prompt, images=[], schema=schema))

    if not requests:
        raise ValueError(
            "No filter requests were built (check hindsight summarize and trajectories)."
        )

    upload_batches_to_bucket(config, requests, bucket, job_name)

    gcs_path_file = gcs_response_path_txt_for_job(config, job_name)
    if not os.path.exists(gcs_path_file):
        print("*" * 100)
        print("Processing job for filter. This may take a while...")
        response_path = process_job(config, job_name)
        with open(gcs_path_file, "w") as f:
            f.write(response_path)
    else:
        print("*" * 100)
        print("Job for filter already processed. Loading response path from file...")
        print(
            "If you'd like to reprocess the job, delete the gcs_response_path.txt file and run again."
        )
        with open(gcs_path_file, "r") as f:
            response_path = f.read().strip("\n")

    process_responses(config, response_path, job_name)
