import os
import wandb
import numpy as np
from typing import List, Optional, Tuple
from cast.atomic_model.train.visualizing.visualize_utils import numpy_to_img
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import clip

def visualize_lang_pred(
    batch_obs_images: np.ndarray,
    batch_lang_preds: str,
    eval_type: str,
    save_folder: str,
    epoch: int,
    num_images_preds: int = 8,
    use_wandb: bool = True,
    display: bool = False,
    rounding: int = 4,
):
    """
    Visualize the distance classification predictions and labels for an observation image pair.

    Args:
        batch_obs_images (np.ndarray): batch of observation images [batch_size, height, width, channels]
        eval_type (string): {data_type}_{eval_type} (e.g. recon_train, gs_test, etc.)
        epoch (int): current epoch number
        num_images_preds (int): number of images to visualize
        use_wandb (bool): whether to use wandb to log the images
        save_folder (str): folder to save the images. If None, will not save the images
        display (bool): whether to display the images
        rounding (int): number of decimal places to round the distance predictions and labels
    """
    visualize_path = os.path.join(
        save_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "dist_classification",
    )
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)

    assert (
        len(batch_obs_images)
        == len(batch_lang_preds)
    )
    batch_size = batch_obs_images.shape[0]
    wandb_list = []
    for i in range(min(batch_size, num_images_preds)):
        obs_image = numpy_to_img(batch_obs_images[i])
        lang_label = batch_lang_preds[i]

        save_path = None
        if save_folder is not None:
            save_path = os.path.join(visualize_path, f"{i}.png")
        text_color = "black"

        display_lang_pred(
            [obs_image],
            ["Observation"],
            lang_label,
            text_color,
            save_path,
            display,
        )
        if use_wandb:
            wandb_list.append(wandb.Image(save_path))
    if use_wandb:
        wandb.log({f"{eval_type}_lang_alignment": wandb_list}, commit=False)

def clip_embed(text, model, device): 
    text = clip.tokenize(text).to(device)
    text_features = model.encode_text(text)
    return text_features

def display_lang_pred(
    imgs: list,
    titles: list,
    lang_label: str,
    text_color: str = "black",
    save_path: Optional[str] = None,
    display: bool = False,
):
    plt.figure()
    fig, ax = plt.subplots(1, len(imgs))
    if len(imgs) == 1:
        ax = [ax]

    plt.suptitle(f"label: {lang_label}", color=text_color)

    for axis, img, title in zip(ax, imgs, titles):
        axis.imshow(img)
        axis.set_title(title)
        axis.xaxis.set_visible(False)
        axis.yaxis.set_visible(False)

    # make the plot large
    fig.set_size_inches((18.5 / 3) * len(imgs), 10.5)

    if save_path is not None:
        fig.savefig(
            save_path,
            bbox_inches="tight",
        )
    if not display:
        plt.close(fig)
