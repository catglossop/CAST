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
from transformers import T5EncoderModel, T5Tokenizer
import requests 
import json
import base64
from io import BytesIO
from google import genai

# ROS
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
from deployment.utils import to_numpy, transform_images, load_model
from cv_bridge import CvBridge

# UTILS
from model.model import ResNetFiLMTransformer
from train.training.train_utils import model_output_diffusion_eval
from train.visualizing.action_utils import plot_trajs_and_points, plot_trajs_and_points_on_image
from train.visualizing.visualize_utils import (
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
ROBOT_CONFIG_PATH ="../../config/robot.yaml"
MODEL_CONFIG_PATH = "../../config/models.yaml"
DATA_CONFIG = "../../data/data_config.yaml"

PRIMITIVES = ["Go forward", "Turn left", "Turn right", "Adjust right", "Adjust left", "Stop"]

class PiAPlanningPolicy(Node): 

    def __init__(self, 
                args
                ):
        super().__init__('navigate_atomic')
        self.args = args
        self.context_queue = []
        self.num_samples = args.num_samples
        self.language_prompt = args.prompt

        # Load the config
        self.load_config(ROBOT_CONFIG_PATH)

        # Load the model 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model_from_config("pia_planning_model.yaml", "pia_planning_model")
        self.language_encoder = self.model_params["language_encoder"]
        self.context_size = self.model_params["context_size"]
        self.img_received = False
        self.naction = None

        # Load language encoder
        self.load_language_encoder(self.language_encoder)
 
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
        api_key = os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        
    def suggest_action(self, image, prompt):
        image.save("current_image.jpg")
        curr_obs_64 = self.image_to_base64(image)
        response_schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": f"Action in the list {PRIMITIVES}",
                    "enum": PRIMITIVES,
                },
                "reasoning": {
                    "type": "string",
                    "description": "Reasoning for the action taken",
                },
            },
            "required": ["action", "reasoning"],
        }
        vlm_prompt = f"A robot is moving through an environment and has the task '{prompt}'. Given the current observation, which action in the list {PRIMITIVES} should the robot take next? Return your response as the single action in the list of primitives with no additional information."
        contents = []
        contents.append({"role": "user",
                                        "parts": [
                                            {
                                                "text": vlm_prompt,
                                            },
                                            {
                                                "inline_data" : {
                                                    "mime_type" : "image/jpeg",
                                                    "data": curr_obs_64,
                                                }
                                            }
                                        ]
                            })
        
        ai_response = self.client.models.generate_content(model="gemini-2.0-flash",
                                                          contents = contents, 
                                                          config={
                                                            'response_mime_type': 'application/json',
                                                            'response_schema': response_schema,
                                                          })
        json_response = json.loads(ai_response.text)
        action = json_response["action"]
        reasoning = json_response["reasoning"]
        print("VLM action: ", action)
        print("VLM reasoning: ", reasoning)
        return action
    
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
            if center_crop:
                if w > h:
                    pil_img = TF.center_crop(pil_img, (h, int(h * IMAGE_ASPECT_RATIO)))  # crop to the right ratio
                else:
                    pil_img = TF.center_crop(pil_img, (int(w / IMAGE_ASPECT_RATIO), w))
            pil_img = pil_img.resize(image_size) 
            transf_img = TF.to_tensor(pil_img)
            transf_img = torch.unsqueeze(transf_img, 0)
            transf_imgs.append(transf_img)
        return torch.cat(transf_imgs, dim=1)

    def compare_output(self):
        dataset_name = "sacson"
        traj_1 = self.nactions
        prompt_1 = self.language_prompt
        viz_img = self.transform_images_viz(self.context_queue[-1], IMAGE_SIZE) 
        fig, ax = plt.subplots(1, 2)
        if len(traj_1.shape) > 2:
            trajs = [*traj_1]
        else:
            trajs = [traj_1]
        start_pos = np.array([0,0])
        goal_pos = np.array([0,0])
        plot_trajs_and_points(
            ax[0], 
            trajs,
            [start_pos, goal_pos], 
            traj_colors=[CYAN] + [MAGENTA]*(self.num_samples - 1),
            point_colors=[GREEN, RED],
            traj_labels = ["gt"] + ["pred"]*(self.num_samples -1)
        )
        plot_trajs_and_points_on_image(      
            ax[1],
            np.transpose(viz_img.numpy().squeeze(0), (1,2,0)),
            dataset_name,
            trajs,
            [start_pos, goal_pos],
            traj_colors=[CYAN] + [MAGENTA]*(self.num_samples - 1),
            point_colors=[GREEN, RED],
        )
        ax[0].legend([prompt_1])
        ax[1].legend([prompt_1])
        ax[0].set_ylim((-5, 5))
        ax[0].set_xlim((-5, 15))
        plt.savefig("visualize.png")

    def load_language_encoder(self, language_encoder):
        print("Loading T5 model")
        self.tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
        self.t5_model = T5EncoderModel.from_pretrained("google-t5/t5-small")

    def embed_language(self, language_prompt):
        language_embedding = self.tokenizer(language_prompt, return_tensors="pt", padding=True)
        language_embedding = self.t5_model(language_embedding["input_ids"]).last_hidden_state.mean(dim=1).to(self.device)
        return language_embedding

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
        
        self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.model_params["num_diffusion_iters"],
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                prediction_type='epsilon'
            )
        self.model = load_model(
            self.ckpth_path,
            self.model_params,
            self.device
        )
        self.model.eval()
    
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
        self.image_msg.save("test_image.jpg")
        if self.context_size is not None:
            if len(self.context_queue) < self.context_size + 1:
                self.context_queue.append(self.image_msg)
            else:
                os.makedirs("context_queue", exist_ok=True)
                for i in range(len(self.context_queue)):
                    self.context_queue[i].save(f"context_queue/{i}.jpg") 
                self.context_queue.pop(0)
                self.context_queue.append(self.image_msg)
        self.img_received = True
    
    def process_images(self):
        self.viz_img = self.context_queue[-1].copy()
        self.viz_img.save("viz_img.jpg")

        self.obs_images = transform_images(self.context_queue, self.model_params["image_size"], center_crop=False)
        self.obs_images = torch.split(self.obs_images, 3, dim=1)
        self.obs_images = torch.cat(self.obs_images, dim=1) 
        self.obs_images = self.obs_images.to(self.device)
        self.mask = torch.zeros(1).long().to(self.device)  

    def image_to_base64(self, image):
        buffer = BytesIO()
        # Convert the image to RGB mode if it's in RGBA mode
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return img_str
    
    def infer_actions(self):

        # First send the image to the VLM to get the next action 
        lang_action = self.suggest_action(self.viz_img, self.language_prompt)
        print("Suggested action: ", lang_action)
        lang_action_embed = self.embed_language(lang_action)

        self.nactions = model_output_diffusion_eval(self.model, 
                                                    self.noise_scheduler, 
                                                    self.obs_images.clone(), 
                                                    lang_action_embed.clone(),
                                                    lang_action, 
                                                    None,
                                                    self.model_params["len_traj_pred"], 
                                                    2, 
                                                    self.num_samples, 
                                                    1, 
                                                    self.device, 
                                                    None)["actions"].detach().cpu().numpy()
        self.sampled_actions_msg = Float32MultiArray()
        self.sampled_actions_msg.data = np.concatenate((np.array([0]), self.nactions.flatten())).tolist()
        self.sampled_actions_pub.publish(self.sampled_actions_msg)
        self.naction = self.nactions[0] 
        self.chosen_waypoint = self.naction[self.args.waypoint] 
        

    def timer_callback(self):
        self.chosen_waypoint = np.zeros(4, dtype=np.float32)
        if len(self.context_queue) > self.context_size and self.img_received:

            # Process observations
            self.process_images()
            
            # Use policy to get actions
            self.infer_actions()

            # Visualize actions 
            self.compare_output()
            self.img_received = False

        # Normalize and publish waypoint
        if self.naction is not None:
            if self.model_params["normalize"]:
                self.naction[:,:2] *= (self.MAX_V / self.RATE)
                self.chosen_waypoint[:2] *= (self.MAX_V / self.RATE)
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

def main(args):
    rclpy.init()
    nav_policy = PiAPlanningPolicy(args)

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
        default=6, # close waypoints exihibit straight line motion (the middle waypoint is a good default)
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
        help="Prompt for the language model",
    )
    args = parser.parse_args()
    main(args)


