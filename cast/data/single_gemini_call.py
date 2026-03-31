import argparse
import json
import os
from glob import glob
from typing import Any, Dict, List

from google import genai
from google.genai import types

from cast.data.utils.common import base_actions, formats


PROMPT_FILE_BY_TASK = {
    "hindsight_describe": "hindsight_describe_prompt.txt",
    "hindsight_summarize": "hindsight_summarize_prompt.txt",
    "filter_step": "filter_step_prompt.txt",
    "cf_step": "cf_step_prompt.txt",
}

SCHEMA_FILE_BY_TASK = {
    "hindsight_describe": "hindsight_describe_schema.json",
    "hindsight_summarize": "hindsight_summarize_schema.json",
    "filter_step": "filter_step_schema.json",
    "cf_step": "cf_step_schema.json",
}


def _sorted_image_paths(traj_path: str) -> List[str]:
    return sorted(
        glob(os.path.join(traj_path, "*.jpg")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
    )


def _pick_single_trajectory(dataset_root: str, explicit_traj_path: str | None) -> str:
    if explicit_traj_path:
        traj_data = os.path.join(explicit_traj_path, "traj_data.pkl")
        if not os.path.exists(traj_data):
            raise FileNotFoundError(f"`traj_data.pkl` not found at `{traj_data}`")
        return explicit_traj_path

    traj_data_files = sorted(glob(os.path.join(dataset_root, "**", "traj_data.pkl"), recursive=True))
    if not traj_data_files:
        raise FileNotFoundError(f"No trajectories found under `{dataset_root}`")
    return os.path.dirname(traj_data_files[0])


def _build_prompt_vars(task: str, traj_path: str, image_paths: List[str]) -> Dict[str, Any]:
    uid = f"{os.path.basename(traj_path.rstrip('/'))}_start_0_end_{max(len(image_paths) - 1, 0)}"
    sample_instructions = [
        "move forward down the corridor",
        "turn slightly left around the obstacle",
    ]
    sample_labels = ["go forward", "turn left"]
    
    descriptions = {}
    with open("/hdd/cast_output/hindsight/hindsight_describe/hindsight_describe_responses.jsonl", "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            path = row["path"]
            if path in image_paths:
                descriptions[path] = row["image_description"]

    variables: Dict[str, Any] = {
        "path": image_paths[min(1, max(len(image_paths) - 1, 0))] if image_paths else "",
        "instructions": "[ " + " ".join(sample_instructions) + " ]",
        "formats": formats,
        "unique_id": uid,
        "labels": "[ " + " ".join(sample_labels) + " ]",
        "orig_instructions": "[ " + " ".join(sample_instructions) + " ]",
        "curr_atomic_action": "go forward",
        "base_actions": "[ " + " ".join(base_actions) + " ]",
        "instructions": descriptions,
    }
    return variables


def _build_contents(task: str, prompt_text: str, image_paths: List[str]) -> List[Dict[str, Any]]:
    parts: List[Any] = [prompt_text]

    if task == "hindsight_describe":
        if len(image_paths) >= 2:
            # Prompt expects previous image first, then current image.
            with open(image_paths[0], 'rb') as f:
                image_bytes = f.read()
            parts.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))
            with open(image_paths[1], 'rb') as f:
                image_bytes = f.read()
            parts.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))
        elif len(image_paths) == 1:
            with open(image_paths[0], 'rb') as f:
                image_bytes = f.read()
            parts.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))
    
    elif task == "cf_step" and image_paths:
        parts.append(image_paths[0])

    return parts


def _load_task_assets(prompt_dir: str, schema_dir: str, task: str) -> tuple[str, Dict[str, Any]]:
    prompt_path = os.path.join(prompt_dir, PROMPT_FILE_BY_TASK[task])
    schema_path = os.path.join(schema_dir, SCHEMA_FILE_BY_TASK[task])

    with open(prompt_path, "r") as f:
        prompt_template = f.read()
    with open(schema_path, "r") as f:
        schema = json.load(f)
        
    return prompt_template, schema


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single Gemini call with CAST prompts/schemas.")
    parser.add_argument(
        "--task",
        choices=list(PROMPT_FILE_BY_TASK.keys()),
        default="hindsight_describe",
        help="Which CAST prompt/schema pair to run.",
    )
    parser.add_argument(
        "--dataset-root",
        default="/hdd/sacson_subset",
        help="Dataset root containing trajectories with traj_data.pkl.",
    )
    parser.add_argument(
        "--trajectory-path",
        default=None,
        help="Optional explicit trajectory path. If omitted, first trajectory is used.",
    )
    parser.add_argument(
        "--prompt-dir",
        default="/hdd/CAST/cast/data/prompts",
        help="Directory containing CAST prompt templates.",
    )
    parser.add_argument(
        "--schema-dir",
        default="/hdd/CAST/cast/data/schemas",
        help="Directory containing CAST response schemas.",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-pro",
        help="Gemini model name.",
    )
    parser.add_argument(
        "--output-path",
        default="/hdd/cast_output/single_gemini/single_call_response.json",
        help="Where to save the raw response and parsed JSON.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set `GOOGLE_API_KEY` or `GEMINI_API_KEY` before running.")

    prompt_template, schema = _load_task_assets(args.prompt_dir, args.schema_dir, args.task)
    traj_path = _pick_single_trajectory(args.dataset_root, args.trajectory_path)
    image_paths = _sorted_image_paths(traj_path)
    prompt_vars = _build_prompt_vars(args.task, traj_path, image_paths)

    try:
        prompt_text = prompt_template.format(**prompt_vars)
    except KeyError as exc:
        missing = str(exc)
        raise KeyError(
            f"Prompt formatting failed for task `{args.task}`. Missing key: {missing}. "
            f"Available keys: {sorted(prompt_vars.keys())}"
        ) from exc

    parts = _build_contents(args.task, prompt_text, image_paths)
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=args.model,
        contents=parts,
        config={
            "response_mime_type": "application/json",
            "response_schema": schema,
        },
    )
    
    breakpoint()

    parsed = None
    if response.text:
        try:
            parsed = json.loads(response.text)
        except json.JSONDecodeError:
            parsed = None

    output = {
        "task": args.task,
        "model": args.model,
        "dataset_root": args.dataset_root,
        "trajectory_path": traj_path,
        "images_used": image_paths[:2] if args.task == "hindsight_describe" else image_paths[:1],
        "prompt_vars": prompt_vars,
        "raw_text": response.text,
        "parsed_json": parsed,
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved response to: {args.output_path}")
    print(f"Task: {args.task}")
    print(f"Trajectory: {traj_path}")
    print(f"Parsed JSON available: {parsed is not None}")


if __name__ == "__main__":
    main()
