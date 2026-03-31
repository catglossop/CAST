import numpy as np
from glob import glob
import os
import pickle as pkl
from PIL import Image
from google import genai
from google.cloud import storage
import json
import time
import tensorflow as tf
from google import genai
from google.genai.types import CreateBatchJobConfig, JobState, HttpOptions, BatchJobDestination, BatchJobSource
from datetime import datetime

formats = "['move from A to B', 'move past A', 'move towards B', 'move away from A', 'move along B', 'move in a C way towards B'] etc."
base_actions = ["turn left", "turn right", "go forward", "stop", "adjust left", "adjust right"]


def setup_gcp(config: dict, force_dataset_reupload: bool = False, incremental_dataset_upload: bool = False) -> None:
    print("*"*100)
    print("Setting up GCP")
    print("*"*100)
    # Setup the bucket and client
    client = storage.Client(project=config["gcp_project_id"])
    if not client.bucket(config["base_uri"]).exists():
        print(f"Bucket {config['base_uri']} does not exist, creating it")
        client.create_bucket(config["base_uri"])

    # Check if the dataset path exists in the bucket
    bucket = client.bucket(config["base_uri"])
    num_blobs = len([b for b in bucket.list_blobs(prefix=config["dataset_path"].split("/")[-1])])
    if num_blobs == 0 or force_dataset_reupload or incremental_dataset_upload:

        if num_blobs > 0 and force_dataset_reupload:
            print(f"Force dataset reupload is True, dataset path {config['dataset_path'].split('/')[-1]} exists in the bucket {config['base_uri']}, deleting it")
            delete_bucket_directory(bucket, config["dataset_path"].split("/")[-1])
        
        print(f"Dataset path {config['dataset_path'].split('/')[-1]} has {num_blobs} blobs, uploading dataset")
        upload_local_data_to_bucket(config)
    else:
        print(f"Dataset path {config['dataset_path'].split('/')[-1]} exists in the bucket {config['base_uri']}")

def get_trajectory_paths(dataset_path: str) -> list:
    """
    Get the paths to all the trajectories in the directory.
    """
    paths = glob(dataset_path + "/**/traj_data.pkl", recursive=True)
    paths = [path.replace("/traj_data.pkl", "") for path in paths]
    return paths

def load_trajectory_data(traj_path: str) -> dict:
    """
    Load the trajectory data from the path.
    """
    data_path = traj_path + "/traj_data.pkl"
    with open(data_path, "rb") as f:
        trajectory_data = pkl.load(f)
    return trajectory_data

def load_image(traj_path: str, image_idx: int) -> Image.Image:
    """
    Load the image from the path.
    """
    image = Image.open(traj_path + f"/{image_idx}.jpg")
    if image.mode != "RGB":
        image = image.convert("RGB") 
    return image

def load_all_images(traj_path: str) -> list:
    """
    Load all the images from the trajectory path.
    """
    images = []
    for image_idx in range(len(glob(traj_path + "/*.jpg"))):
        images.append(load_image(traj_path, image_idx))
    return images

def load_all_images_paths(traj_path: str) -> list:
    """
    Load all the images from the trajectory path.
    """
    image_paths = sorted(glob(traj_path + "/*.jpg"), key=lambda x: int(x.split("/")[-1].split(".")[0]))
    return image_paths

def initialize_gemini_client(config: dict) -> None:
    """
    Initialize the Gemini client.
    """
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    client = genai.GenerativeModel(model_name=config["model_name"])
    return client

def fill_in_prompt(prompt: str, data: dict) -> str:
    """
    Fill in the prompt with the data.
    """
    return prompt.format(**data)

def make_request(prompt: str, images: list, schema: dict) -> None:
    """
    Make the request to the Gemini API.
    """
    parts = [{"text": prompt}]
    for img in images:
        if isinstance(img, str) and img.startswith("gs://"):
            parts.append({"fileData": {"fileUri": img, "mimeType": "image/jpeg"}})
        else:
            parts.append({"inlineData": {"data": img, "mimeType": "image/jpeg"}})

    contents = [{"role": "user", "parts": parts}]
    request = {
        "request": {
            "contents": contents,
            "generationConfig": {
                "responseSchema": schema,
                "responseMimeType": "application/json",
            },
        }
    }

    return request

def upload_images_to_bucket(config: dict, image_data: list, bucket: storage.Bucket, prefix: str) -> None:
    """
    Upload the images to the bucket.
    """
    uris = []
    for img_path in image_data:
        # Get the dataset 
        rel_path = os.path.relpath(img_path, config["dataset_path"])
        blob = bucket.blob(f"{prefix}/{rel_path}")
        blob.upload_from_filename(img_path)
        uris.append(f"gs://{bucket.name}/{prefix}/{rel_path}")
    return uris

def convert_path_to_uri(config: dict, path: str) -> str:
    """
    Convert the path to the URI.
    """
    rel_path = os.path.relpath(path, config["dataset_path"])
    return f"gs://{config['base_uri']}/{config['dataset_path'].split('/')[-1]}/{rel_path}"

def upload_local_data_to_bucket(config: dict) -> None:
    """
    Upload the local data to the bucket.
    """
    client = storage.Client(project=config["gcp_project_id"])
    bucket = client.bucket(config["base_uri"])
    dataset_name = config["dataset_path"].split("/")[-1]
    files = [file for file in glob(config["dataset_path"] + "/**/*", recursive=True) if os.path.isfile(file)]
    for file in files:
        rel_path = os.path.relpath(file, config["dataset_path"])
        blob = bucket.blob(f"{dataset_name}/{rel_path}")
        if not blob.exists():
            blob.upload_from_filename(file)
        else:
            print(f"Blob {blob.name} already exists, skipping upload")

def delete_bucket_directory(bucket: storage.Bucket, directory: str) -> None:
    """
    Delete the directory from the bucket.
    """
    for blob in bucket.list_blobs(prefix=directory):
        blob.delete()

def upload_batches_to_bucket(config: dict, requests: list, bucket: storage.Bucket, job_name: str) -> None:
    """
    Upload the requests to the bucket.
    """
    blob = bucket.blob(f"{config['blob_root']}_{job_name}_requests/requests.jsonl")
    blob.upload_from_string("\n".join([json.dumps(request) for request in requests]), content_type="application/jsonl")

def process_job(config: dict, job_name : str) -> str: 
    input_uri = f"gs://{config['base_uri']}/{config['blob_root']}_{job_name}_requests/requests.jsonl"
    output_uri = f"gs://{config['base_uri']}/{config['blob_root']}_{job_name}_responses/responses_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"

    client = genai.Client(project=config["gcp_project_id"], location=config["gcp_region"], http_options=HttpOptions(api_version="v1"))

    batch_prediction_job = client.batches.create(
        model=config["model_version"],
        src=BatchJobSource(gcs_uri=[input_uri], format="jsonl"),
        config=CreateBatchJobConfig(dest=BatchJobDestination(gcs_uri=output_uri, format="jsonl")),
    )
    print(f"Job name: {batch_prediction_job.name}")
    print(f"Job state: {batch_prediction_job.state}")
    completed_states = {
        JobState.JOB_STATE_SUCCEEDED,
        JobState.JOB_STATE_FAILED,
        JobState.JOB_STATE_CANCELLED,
        JobState.JOB_STATE_PAUSED,
    }

    while batch_prediction_job.state not in completed_states:
        time.sleep(30)
        batch_prediction_job = client.batches.get(name=batch_prediction_job.name)
        print(f"Job state: {batch_prediction_job.state}")

    if getattr(batch_prediction_job.state, "name", str(batch_prediction_job.state)) != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Batch job ended with state {batch_prediction_job.state}")

    # Vertex batch output is written under the destination prefix (predictions.jsonl).
    bucket = storage.Client(project=config["gcp_project_id"]).bucket(config["base_uri"])
    blobs = bucket.list_blobs(prefix=batch_prediction_job.dest.gcs_uri.lstrip(f"gs://").split(f"{config['base_uri']}/")[-1])
    for blob in blobs:
        if blob.name.endswith("predictions.jsonl"):
            output_path = f"gs://{config['base_uri']}/{blob.name}"
            break

    return output_path

def process_responses(config: dict, response_path: str, job_name : str) -> None:
    """
    Process the responses from the job.
    """
    errors = 0
    total_responses = 0
    responses = []
    with tf.io.gfile.GFile(response_path, "r") as f:

        for line in f:
            total_responses += 1
            response = json.loads(line)
            if response["status"] != "":
                errors += 1 
            else:
                try: 
                    response_json = json.loads(response["response"]["candidates"][0]["content"]["parts"][0]["text"])
                    responses.append(response_json)
                except:
                    errors += 1 
    
    save_response(config, responses, job_name)
    print("*"*100)
    print(f"Job {job_name} completed with {errors} errors out of {total_responses} requests")
    print("*"*100)

def gcs_response_path_txt_for_job(config: dict, job_name: str) -> str:
    """
    Local path to the text file storing the GCS predictions.jsonl URI for a batch job.
    Matches the layout used by hindsight steps: {output_dir}/{job_root}/{job_name}/gcs_response_path.txt
    """
    job_root = job_name.split("_")[0]
    return os.path.join(config["output_dir"], job_root, job_name, "gcs_response_path.txt")


def saved_batch_responses_path(config: dict, job_name: str) -> str:
    """
    Local path where process_responses / save_response write parsed model outputs.
    """
    job_root = job_name.split("_")[0]
    d = os.path.join(config["output_dir"], job_root, job_name)
    return os.path.join(d, f"{job_name}_responses.jsonl")


def save_response(config: dict, responses: dict, job_name : str) -> None:
    """
    Save the response to the local directory.
    """
    job_root = job_name.split("_")[0]
    output_dir = f"{config['output_dir']}/{job_root}/{job_name}"
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/{job_name}_responses.jsonl", "w") as f:
        f.write("\n".join([json.dumps(response) for response in responses]))

def join_string_list(string_list: list) -> str:
    """
    Join the instructions into a single string.
    """
    return "[ " + " ".join(string_list) + " ]"

