
import os
import sys
import io
import matplotlib.pyplot as plt

# ROS
from sensor_msgs.msg import Image

# pytorch
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as TF

import numpy as np
from PIL import Image as PILImage
from typing import List, Tuple, Dict, Optional
from prettytable import PrettyTable 

# models
from model.model import ResNetFiLMTransformer
from model.lelan.lnp_comp import LNP_comp, LNP_clip_FiLM, LNPMultiModal
from model.lelan.lnp import LNP_clip, LNP, DenseNetwork_lnp, LNP_MM
from model.nomad.nomad import NoMaD, DenseNetwork
from model.nomad.nomad_vint import NoMaD_ViNT
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from train.training.train_utils import replace_bn_with_gn, model_output_diffusion_eval
from data.data_utils import IMAGE_ASPECT_RATIO


def load_model(
    model_path: str,
    config: dict,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Load a model from a checkpoint file (works with models trained on multiple GPUs)"""
    model_type = config["model_type"]
    
    if model_type == "rft":
        model = ResNetFiLMTransformer(
            config["efficientnet_model"],
            config["context_size"],
            config["len_traj_pred"],
            config["encoding_size"],
            config["lang_encoding_size"],
            config["mha_num_attention_layers"],
            config["mha_num_attention_heads"],
            config["vocab_size"],
            config["dropout"],
            device,
        )
        checkpoint = torch.load(model_path, map_location=device)
    elif model_type == "lnp" or model_type == "upweight":
        if config["vision_encoder"] == "lnp_clip_film":
            vision_encoder = LNP_clip_FiLM(
                obs_encoder=config["obs_encoder"],
                obs_encoding_size=config["obs_encoding_size"],
                lang_encoding_size=config["lang_encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
                )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        noise_scheduler = DDPMScheduler(
                num_train_timesteps=config["num_diffusion_iters"],
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                prediction_type='epsilon'
            )
        if model_type == "upweight":
            noise_pred_net = ConditionalUnet1D(
                    input_dim=2,
                    global_cond_dim=config["encoding_size"]*2,
                    down_dims=config["down_dims"],
                    cond_predict_scale=config["cond_predict_scale"],
                )
        else:
                noise_pred_net = ConditionalUnet1D(
                    input_dim=2,
                    global_cond_dim=config["encoding_size"],
                    down_dims=config["down_dims"],
                    cond_predict_scale=config["cond_predict_scale"],
                )
        dist_pred_network = DenseNetwork_lnp(embedding_dim=config["encoding_size"]*(config["context_size"]+1), control_horizon=config["len_traj_pred"])
        model = LNP_clip(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        )
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(0)) 
    elif config["model_type"] == "lnp_multi_modal":
        if config["vision_encoder"] == "lnp_multi_modal":
            vision_encoder = LNPMultiModal(
                obs_encoder=config["obs_encoder"],
                obs_encoding_size=config["obs_encoding_size"],
                lang_encoding_size=config["lang_encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
                late_fusion=config["late_fusion"],
                per_obs_film=config["per_obs_film"],
                use_film=config["use_film"],
                use_transformer=config["use_transformer"],
                )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        if config["action_head"] == "diffusion":
            noise_scheduler = DDPMScheduler(
                    num_train_timesteps=config["num_diffusion_iters"],
                    beta_schedule='squaredcos_cap_v2',
                    clip_sample=True,
                    prediction_type='epsilon'
                )
            if config["categorical"]:
                action_head = ConditionalUnet1D(
                    input_dim=2,
                    global_cond_dim=4,
                    down_dims=config["down_dims"],
                    cond_predict_scale=config["cond_predict_scale"],
                )
            else:
                action_head = ConditionalUnet1D(
                        input_dim=2,
                        global_cond_dim=config["encoding_size"],
                        down_dims=config["down_dims"],
                        cond_predict_scale=config["cond_predict_scale"],
                    )
            dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
        elif config["action_head"] == "dense":
            noise_scheduler = None
            if config["categorical"]:
                action_head = DenseNetwork_lnp(embedding_dim=4, control_horizon=config["len_traj_pred"])
                dist_pred_network = None
            else:
                action_head = DenseNetwork_lnp(embedding_dim=config["encoding_size"], control_horizon=config["len_traj_pred"])
                dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
        model = LNP_MM(
            vision_encoder=vision_encoder,
            action_head=action_head,
            dist_pred_net=dist_pred_network,
            action_head_type=config["action_head"],
        ) 
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(0)) 
    elif config["model_type"] == "nomad":
        if config["vision_encoder"] == "nomad_vint":
            vision_encoder = NoMaD_ViNT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        elif config["vision_encoder"] == "vit": 
            vision_encoder = ViT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                image_size=config["image_size"],
                patch_size=config["patch_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        
        noise_pred_net = ConditionalUnet1D(
                input_dim=2,
                global_cond_dim=config["encoding_size"],
                down_dims=config["down_dims"],
                cond_predict_scale=config["cond_predict_scale"],
            )
        dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
        
        model = NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        )

        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(0))
        
    if model_type == "lnp" or model_type == "nomad" or model_type == "upweight" or model_type == "lnp_multi_modal":
        state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        loaded_model = checkpoint["model"]
        try:
            state_dict = loaded_model.module.state_dict()
            model.load_state_dict(state_dict, strict=False)
        except AttributeError as e:
            state_dict = loaded_model.state_dict()
            model.load_state_dict(state_dict, strict=False)
    
    num_params = count_parameters(model)
    print("Number of parameters: ",num_params)

            
    return model.to(device)

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    # print(table)
    print(f"Total Trainable Params: {total_params/1e6:.2f}M")
    return total_params

# def msg_to_pil(msg: Image) -> PILImage.Image:
#     img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
#         msg.height, msg.width, -1)
#     pil_image = PILImage.fromarray(img)
#     return pil_image


# def pil_to_msg(pil_img: PILImage.Image, encoding="mono8") -> Image:
#     img = np.asarray(pil_img)  
#     ros_image = Image(encoding=encoding)
#     ros_image.height, ros_image.width, _ = img.shape
#     ros_image.data = img.ravel().tobytes() 
#     ros_image.step = ros_image.width
#     return ros_image


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def transform_images(pil_imgs: List[PILImage.Image], image_size: List[int], center_crop: bool = False) -> torch.Tensor:
    """Transforms a list of PIL image to a torch tensor."""
    transform_type = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                    0.229, 0.224, 0.225]),
        ]
    )
    if type(pil_imgs) != list:
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        w, h = pil_img.size
        if center_crop:
            if w > h:
                pil_img = TF.center_crop(pil_img, (h, int(h * IMAGE_ASPECT_RATIO)))  # crop to the right ratio
            else:
                pil_img = TF.center_crop(pil_img, (int(w / IMAGE_ASPECT_RATIO), w))
        pil_img = pil_img.resize(image_size) 
        transf_img = transform_type(pil_img)
        transf_img = torch.unsqueeze(transf_img, 0)
        transf_imgs.append(transf_img)
    return torch.cat(transf_imgs, dim=1)
    

# clip angle between -pi and pi
def clip_angle(angle):
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi
