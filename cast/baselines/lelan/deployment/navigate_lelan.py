import matplotlib.pyplot as plt
import os
from typing import Tuple, Sequence, Dict, Union, Optional, Callable, List
import numpy as np
import yaml
import threading
from PIL import Image as PILImage
import argparse
import time
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import clip 
from torchvision import transforms
import torchvision.transforms.functional as TF
import tensorflow_hub as hub
# import tensorflow_text
from transformers import T5EncoderModel, T5Tokenizer

# Model 
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from baselines.lelan.train.vint_train.models.lelan.lelan import LeLaN_clip_BC
from baselines.lelan.train.vint_train.models.lelan.lelan_comp import LeLaN_clip_FiLM
from baselines.lelan.train.vint_train.models.nomad.nomad import DenseNetwork
from baselines.lelan.train.vint_train.models.nomad.nomad_vint import replace_bn_with_gn

# ROS
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
from deployment.utils import to_numpy, transform_images
from cv_bridge import CvBridge

# UTILS
from model.model import ResNetFiLMTransformer
from baselines.lelan.train.vint_train.training.train_utils_lelan import model_output_lelan_lcbc
from baselines.lelan.train.vint_train.visualizing.action_utils import plot_trajs_and_points, plot_trajs_and_points_on_image
from baselines.lelan.train.vint_train.visualizing.visualize_utils import (
    to_numpy,
    numpy_to_img,
    VIZ_IMAGE_SIZE,
    RED,
    GREEN,
    BLUE,
    CYAN,
    YELLOW,
    MAGENTA,
)
IMAGE_SIZE = (96, 96)
from data.data_utils import IMAGE_ASPECT_RATIO
from deployment.topic_names import (IMAGE_TOPIC,
                        WAYPOINT_TOPIC,
                        SAMPLED_ACTIONS_TOPIC, 
                        REACHED_GOAL_TOPIC)

# CONSTANTS
ROBOT_CONFIG_PATH ="../../../config/robot.yaml"
MODEL_CONFIG_PATH = "../../../config/models.yaml"
DATA_CONFIG = "../../../data/data_config.yaml"

class NavigateLeLaN(Node): 

    def __init__(self, 
                args
                ):
        super().__init__('navigate_lelan')
        self.args = args
        self.context_queue = []
        self.num_samples = args.num_samples

        transform = ([
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
        self.transform = transforms.Compose(transform)
        
        self.language_prompt = args.prompt

        # Load the config
        self.load_config(ROBOT_CONFIG_PATH)

        # Load the model 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model_from_config("lelan_model.yaml", "lelan")
        self.context_size = self.model_params["context_size"]
 
        # Load data config
        self.load_data_config()
        
        # SUBSCRIBERS  
        self.image_msg = Image()
        self.image_sub = self.create_subscription(
            Image,
            IMAGE_TOPIC,
            self.image_callback,
            1)
        
        # PUBLISHERS
        self.sampled_actions_msg = Float32MultiArray()
        self.sampled_actions_pub = self.create_publisher(
            Float32MultiArray, 
            SAMPLED_ACTIONS_TOPIC, 
            1)
        self.waypoint_msg = Float32MultiArray()
        self.waypoint_pub = self.create_publisher(
            Float32MultiArray, 
            WAYPOINT_TOPIC, 
            1)  
        self.bridge = CvBridge()
        
        # TIMERS
        self.timer_period = 1/self.RATE  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
    
    # Utils
    def unnormalize_data(self, ndata, stats):
        ndata = (ndata + 1) / 2
        data = ndata * (stats['max'] - stats['min']) + stats['min']
        return data

    def get_delta(self, actions):
        # append zeros to first action
        ex_actions = np.concatenate([np.zeros((actions.shape[0],1,actions.shape[-1])), actions], axis=1)
        delta = ex_actions[:,1:] - ex_actions[:,:-1]
        return delta

    def get_action(self):
        # diffusion_output: (B, 2*T+1, 1)
        # return: (B, T-1)
        ndeltas = self.nactions
        ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
        ndeltas = to_numpy(ndeltas)
        ndeltas = self.unnormalize_data(ndeltas, self.ACTION_STATS)
        actions = np.cumsum(ndeltas, axis=1)
        return torch.from_numpy(actions).to(self.device)
    
    def transform_images_viz(self, pil_imgs: List[PILImage.Image], image_size: List[int], center_crop: bool = False) -> torch.Tensor:
        """Transforms a list of PIL image to a torch tensor."""
        if type(pil_imgs) != list:
            pil_imgs = [pil_imgs]
        transf_imgs = []
        for pil_img in pil_imgs:
            w, h = pil_img.size
            if w > h:
                pil_img = TF.center_crop(pil_img, (h, int(h * IMAGE_ASPECT_RATIO)))  # crop to the right ratio
            else:
                pil_img = TF.center_crop(pil_img, (int(w / IMAGE_ASPECT_RATIO), w))
            pil_img = pil_img.resize(image_size) 
            transf_img = TF.to_tensor(pil_img)
            transf_img = torch.unsqueeze(transf_img, 0)
            transf_imgs.append(transf_img)
        return torch.cat(transf_imgs, dim=1)

    def load_config(self, robot_config_path):
        with open(robot_config_path, "r") as f:
            robot_config = yaml.safe_load(f)
        self.MAX_V = robot_config["max_v"]
        self.MAX_W = robot_config["max_w"]
        self.VEL_TOPIC = "/task_vel"
        self.DT = 1/robot_config["frame_rate"]
        self.RATE = robot_config["frame_rate"]
        self.EPS = 1e-8
        self.WAYPOINT_TIMEOUT = 1 # seconds # TODO: tune this
        self.FLIP_ANG_VEL = np.pi/4
    
    def load_model_from_config(self, model_paths_config, model_type):
        # Load configs
        with open(model_paths_config, "r") as f:
            model_paths = yaml.safe_load(f)

        model_config_path = model_paths[model_type]["config_path"]
        with open(model_config_path, "r") as f:
            self.model_params = yaml.safe_load(f)
            
        # Load model weights
        self.ckpth_path = model_paths[model_type]["ckpt_path"]
        if os.path.exists(self.ckpth_path):
            print(f"Loading model from {self.ckpth_path}")
        else:
            raise FileNotFoundError(f"Model weights not found at {self.ckpth_path}")
        
        ## LOAD THE LELAN MODEL
        # Load the vision encoder
        vision_encoder = LeLaN_clip_FiLM(
            obs_encoding_size=self.model_params["encoding_size"],
            context_size=self.model_params["context_size"],
            mha_num_attention_heads=self.model_params["mha_num_attention_heads"],
            mha_num_attention_layers=self.model_params["mha_num_attention_layers"],
            mha_ff_dim_factor=self.model_params["mha_ff_dim_factor"],
            feature_size=self.model_params["feature_size"],
            clip_type=self.model_params["clip_type"],
        )
        vision_encoder = replace_bn_with_gn(vision_encoder)   

        # Load the text encoder    
        text_encoder, preprocess = clip.load(self.model_params["clip_type"]) #, device=device    
        text_encoder.to(torch.float32)   
        
        # Load the noise prediction network
        noise_pred_net = ConditionalUnet1D(
                input_dim=2,
                global_cond_dim=self.model_params["encoding_size"],
                down_dims=self.model_params["down_dims"],
                cond_predict_scale=self.model_params["cond_predict_scale"],
            )
        
        # Load dist pred network
        dist_pred_network = DenseNetwork(embedding_dim=self.model_params["encoding_size"])

        self.model = LeLaN_clip_BC(
            vision_encoder=vision_encoder,
            dist_pred_net=dist_pred_network,
            text_encoder=text_encoder,
            noise_pred_net=noise_pred_net,
        )  

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.model_params["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )  
        self.model.to(self.device)
    
    def load_data_config(self):
        # LOAD DATA CONFIG
        with open(os.path.join(os.path.dirname(__file__), DATA_CONFIG), "r") as f:
            data_config = yaml.safe_load(f)
        # POPULATE ACTION STATS
        self.ACTION_STATS = {}
        for key in data_config['action_stats']:
            self.ACTION_STATS[key] = np.array(data_config['action_stats'][key])

    def image_callback(self, msg):
        self.image_msg = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        self.image_msg = PILImage.fromarray(self.image_msg)
        img = self.image_msg.save("test_image.jpg")
        if self.context_size is not None:
            if len(self.context_queue) < self.context_size + 1:
                self.context_queue.append(self.image_msg)
            else:
                os.makedirs("context_queue", exist_ok=True)
                for i in range(len(self.context_queue)):
                    self.context_queue[i].save(f"context_queue/{i}.jpg") 
                self.context_queue.pop(0)
                self.context_queue.append(self.image_msg)

    def process_images(self):
        self.obs_images = transform_images(self.context_queue, self.model_params["image_size"], center_crop=False)
        self.obs_images = torch.split(self.obs_images, 3, dim=1)
        self.obs_images = torch.cat(self.obs_images, dim=1) 
        self.obs_images = self.obs_images.to(self.device)
        self.mask = torch.zeros(1).long().to(self.device)  
    
    def infer_actions(self):
        self.nactions = model_output_lelan_lcbc(self.model, 
                                                self.noise_scheduler, 
                                                self.obs_images.clone(),
                                                self.language_prompt,
                                                self.transform,
                                                self.model_params["len_traj_pred"],
                                                2,
                                                self.num_samples,
                                                self.device).detach().cpu().numpy()
        self.nactions = np.cumsum(self.nactions, axis=0)
        self.sampled_actions_msg = Float32MultiArray()
        self.sampled_actions_msg.data = np.concatenate((np.array([0]), self.nactions.flatten())).tolist()
        print("Sampled actions shape: ", self.nactions.shape)
        self.sampled_actions_pub.publish(self.sampled_actions_msg)
        self.naction = self.nactions[0] 
        self.chosen_waypoint = self.naction[self.args.waypoint] 
        

    def timer_callback(self):
        start = time.time()
        self.chosen_waypoint = np.zeros(4, dtype=np.float32)
        if len(self.context_queue) > self.context_size:

            # Process observations
            self.process_images()
            
            # Use policy to get actions
            self.infer_actions()

            # Visualize actions 
            self.compare_output()

            # Normalize and publish waypoint
            if self.naction is not None:
                if self.model_params["normalize"]:
                    self.naction[:,:2] *= (self.MAX_V / self.RATE)
                    self.chosen_waypoint[:2] *= (self.MAX_V / self.RATE)
                print("Chosen waypoint shape: ", self.chosen_waypoint.shape)
                print("Chosen waypoint: ", self.chosen_waypoint)
                self.execute = 0
                while self.execute < self.args.waypoint:
                    self.waypoint = self.naction[self.execute, :]
                    self.waypoint_msg.data = self.waypoint.tolist()
                    self.waypoint_pub.publish(self.waypoint_msg)
                    time.sleep(self.timer_period)
                    self.execute += 1
                time.sleep(self.timer_period)
                self.blank_msg = Float32MultiArray()
                self.blank_msg.data = np.zeros(4, dtype=np.float32).tolist()
                self.waypoint_pub.publish(self.blank_msg)
            self.naction = None
            print("Elapsed time: ", time.time() - start )

def main(args):
    rclpy.init()
    nav_policy = NavigateLeLaN(args)

    rclpy.spin(nav_policy)
    nav_policy.destroy_node()
    rclpy.shutdown()
    
    print("Registered with master node. Waiting for image observations...")
    print(f"Using {nav_policy.device}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to run local language conditioned navigation")
    parser.add_argument(
        "--waypoint",
        "-w",
        default=2, # close waypoints exihibit straight line motion (the middle waypoint is a good default)
        type=int,
        help=f"""index of the waypoint used for navigation (between 0 and 4 or 
        how many waypoints your model predicts) (default: 2)""",
    )
    parser.add_argument("--num-samples",
        "-n",
        default=1,
        type=int,
        help=f"Number of actions sampled from the exploration model (default: 8)",
    )
    parser.add_argument(
        "--prompt",
        "-p", 
        default="Go to the kitchen",
        type=str,
        help="Prompt for the policy",
    )
    args = parser.parse_args()
    main(args)


