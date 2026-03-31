"""Project trajectory onto the first camera image for filter / counterfactual prompts."""
import os
import numpy as np
from typing import Optional
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from cast.data.utils.common import load_trajectory_data

CAMERA_METRICS = {
    "camera_height": 0.95,
    "camera_x_offset": 0.45,
    "camera_matrix": {"fx": 272.547000, "fy": 266.358000, "cx": 320.000000, "cy": 220.000000},
    "dist_coeffs": {"k1": -0.038483, "k2": -0.010456, "p1": 0.003930, "p2": -0.001007, "k3": 0.000000},
}
VIZ_IMAGE_SIZE = (480, 640)  # (height, width)


def project_points(
    xy: np.ndarray,
    camera_height: float,
    camera_x_offset: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
):
    batch_size, horizon, _ = xy.shape
    xyz = np.concatenate(
        [xy, -camera_height * np.ones(list(xy.shape[:-1]) + [1])], axis=-1
    )
    rvec = tvec = (0, 0, 0)
    xyz[..., 0] += camera_x_offset
    xyz_cv = np.stack([xyz[..., 1], -xyz[..., 2], xyz[..., 0]], axis=-1)
    uv, _ = cv2.projectPoints(
        xyz_cv.reshape(batch_size * horizon, 3), rvec, tvec, camera_matrix, dist_coeffs
    )
    return uv.reshape(batch_size, horizon, 2)


def get_pos_pixels(
    points: np.ndarray,
    camera_height: float,
    camera_x_offset: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
):
    pixels = project_points(
        points[np.newaxis], camera_height, camera_x_offset, camera_matrix, dist_coeffs
    )[0]
    pixels[:, 0] = VIZ_IMAGE_SIZE[0] - pixels[:, 0]
    return np.array(
        [
            p
            for p in pixels
            if np.all(p > 0) and np.all(p < [VIZ_IMAGE_SIZE[0], VIZ_IMAGE_SIZE[1]])
        ]
    )


def draw_trajectory(path: str, start: int, end: int) -> Optional[Image.Image]:
    """Overlay planned path on the image at `start`; trajectory from odom [start:end]."""
    import pickle as pkl

    with open(os.path.join(path, "traj_data.pkl"), "rb") as f:
        traj_data = pkl.load(f)
    curr_traj_data = {key: traj_data[key][start:end] for key in traj_data.keys()}

    odom = np.hstack((curr_traj_data["position"], curr_traj_data["yaw"].reshape(-1, 1)))
    odom = odom - odom[0, :]

    img = Image.open(os.path.join(path, f"{start}.jpg"))
    img = img.resize((VIZ_IMAGE_SIZE[1], VIZ_IMAGE_SIZE[0]))
    fig, ax = plt.subplots()
    ax.imshow(img)

    camera_height = CAMERA_METRICS["camera_height"]
    camera_x_offset = CAMERA_METRICS["camera_x_offset"]
    fx = CAMERA_METRICS["camera_matrix"]["fx"]
    fy = CAMERA_METRICS["camera_matrix"]["fy"]
    cx = CAMERA_METRICS["camera_matrix"]["cx"]
    cy = CAMERA_METRICS["camera_matrix"]["cy"]
    camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    k1 = CAMERA_METRICS["dist_coeffs"]["k1"]
    k2 = CAMERA_METRICS["dist_coeffs"]["k2"]
    p1 = CAMERA_METRICS["dist_coeffs"]["p1"]
    p2 = CAMERA_METRICS["dist_coeffs"]["p2"]
    k3 = CAMERA_METRICS["dist_coeffs"]["k3"]
    dist_coeffs = np.array([k1, k2, p1, p2, k3, 0.0, 0.0, 0.0])

    xy_coords = odom[:, :2]
    traj_pixels = get_pos_pixels(
        xy_coords, camera_height, camera_x_offset, camera_matrix, dist_coeffs
    )
    if len(traj_pixels.shape) != 2:
        plt.close(fig)
        return None

    ax.plot(
        traj_pixels[:250, 0],
        traj_pixels[:250, 1],
        color="blue",
        lw=2.5,
    )
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_xlim((0.5, VIZ_IMAGE_SIZE[1] - 0.5))
    ax.set_ylim((VIZ_IMAGE_SIZE[0] - 0.5, 0.5))
    out_path = os.path.join(path, "_cast_traj_viz_tmp.jpg")
    plt.savefig(out_path)
    plt.close(fig)
    out_img = Image.open(out_path).convert("RGB")
    os.remove(out_path)
    return out_img


def save_trajectory_viz_for_batch(path: str, out_name: str = "cast_trajectory_viz.jpg") -> str:
    """Full-horizon overlay; returns path to written JPEG under `path`."""
    traj_data = load_trajectory_data(path)
    n = len(traj_data["position"])
    if n < 2:
        raise ValueError(f"Not enough frames in {path}")
    viz = draw_trajectory(path, 0, n)
    if viz is None:
        raise RuntimeError(f"Could not project trajectory for {path}")
    out = os.path.join(path, out_name)
    viz.save(out, quality=95)
    return out
