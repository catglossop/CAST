import numpy as np
import pickle as pkl
from cast.data.utils.common import load_trajectory_data, get_trajectory_paths

def get_yaw_delta(yaw_1, yaw_2):
    # Get the two cases of yaw delta
    yaw_delta_init = yaw_2 - yaw_1
    if (yaw_delta_init > 0):
        yaw_delta_wrap = yaw_delta_init - 2*np.pi
    else:
        yaw_delta_wrap = yaw_delta_init + 2*np.pi
    yaw_delta = yaw_delta_init if np.abs(yaw_delta_init) < np.abs(yaw_delta_wrap) else yaw_delta_wrap
    return yaw_delta

def discretize_trajectory(path, config):

    # Load data
    traj_data = load_trajectory_data(path)
    traj_len = len(traj_data["position"])

    # Get position in traj frame
    pos = traj_data["position"] - traj_data["position"][0]
    if traj_len <= 1: 
        return []

    # Compute yaw in traj frame
    yaw = traj_data["yaw"] - traj_data["yaw"][0]

    # Loop through traj to get atomic actions
    idx = 0
    segments = []
    while idx < traj_len:

        # Initialize current trajectory length
        curr_traj_len = 1
        
        # Get chunk of trajectory until turn occurs or max steps reached
        while (idx + curr_traj_len < traj_len
            and np.abs(get_yaw_delta(yaw[idx], yaw[idx + curr_traj_len])) < config["min_turn_thres"] 
            and curr_traj_len <= config["max_atomic_chunk_length"]):
            curr_traj_len += 1

        end_idx = idx + curr_traj_len if idx + curr_traj_len < traj_len else -1

        # Get yaw delta and distance
        yaw_delta = get_yaw_delta(yaw[idx], yaw[end_idx])
        distance = np.sqrt(np.sum(np.square(pos[end_idx,:] - pos[idx,:]), axis=-1))

        # Check if the chunk is turning left
        if yaw_delta > config["min_turn_thres"]:
            label = "turn left"

        # Check if the chunk is turning right   
        elif yaw_delta < -config["min_turn_thres"]:
            label = "turn right"

        # Check if the chunk is going forward
        elif distance > config["min_forward_thres"] and np.abs(yaw_delta) < config["min_adjust_thres"]:
            label = "go forward"
        
        # Check if the chunk is stopping
        elif distance < config["min_forward_thres"]:
            label = "stop"

        # Check if the chunk is adjusting
        elif config["min_adjust_thres"] <= np.abs(yaw_delta) <= config["max_adjust_thres"]:
            if yaw_delta > 0:
                label = "adjust left"

            else:
                label = "adjust right"
        else:
            idx += curr_traj_len
            continue
        
        # Get traj data for current chunk
        atomic_traj_data = {}
        atomic_traj_data["label"] = label
        atomic_traj_data["start"] = idx
        atomic_traj_data["end"] = end_idx
        atomic_traj_data["path"] = path
        segments.append(atomic_traj_data)
        
        idx += curr_traj_len

    return segments