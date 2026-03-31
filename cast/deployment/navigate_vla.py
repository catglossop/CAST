import matplotlib.pyplot as plt
import os
from typing import Tuple, Sequence, Dict, Union, Optional, Callable, List
import numpy as np
import yaml
import threading
from PIL import Image as PILImage
import argparse
import time
import torchvision.transforms.functional as TF
import requests
from io import BytesIO
import base64
import cv2
import pickle as pkl

# ROS
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
from cv_bridge import CvBridge

# UTILS
from data.data_utils import IMAGE_ASPECT_RATIO
from topic_names import (IMAGE_TOPIC,
                        WAYPOINT_TOPIC,
                        SAMPLED_ACTIONS_TOPIC, 
                        REACHED_GOAL_TOPIC)
# CONSTANTS
ROBOT_CONFIG_PATH ="../config/robot.yaml"
DATA_CONFIG = "../data/data_config.yaml"
IMAGE_SIZE = (224, 224)
NORMALIZE = True

class NavigateVLA(Node): 

    def __init__(self, 
                args
                ):
        super().__init__('navigate_vla')
        self.args = args
        self.context_queue = []
        self.context_size = 1
        self.language_prompt = args.prompt
        self.server_address = args.server_address
        self.num_samples = args.num_samples
        self.step = 0

        # Load the config
        self.load_config(ROBOT_CONFIG_PATH)
 
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

    # Utility functions
    def image_to_base64(self, image):
        buffer = BytesIO()
        # Convert the image to RGB mode if it's in RGBA mode
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return img_str

    def transform_images_vla(self, pil_imgs: List[PILImage.Image], image_size: List[int], center_crop: bool = False):
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
            transf_img = np.array(pil_img)
            transf_img = np.expand_dims(transf_img, axis=0)
            transf_imgs.append(transf_img)
        return np.concatenate(transf_imgs, axis=0)

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
        self.obs_images = self.transform_images_vla(self.context_queue, IMAGE_SIZE, center_crop=True)
    
    def infer_actions(self):
        obs_base64 = self.image_to_base64(PILImage.fromarray(np.array(self.context_queue[-1])))
        req_str = self.server_address + str("/gen_action")
        response = requests.post(req_str, json={'obs': obs_base64, 'prompt':self.language_prompt}, timeout=99999999)
        ndeltas = np.array(response.json()['action'])
        ndeltas = ndeltas.reshape(-1, 2)
        self.nactions = ndeltas
        self.nactions = np.cumsum(ndeltas, axis=0)
        self.nactions -= self.nactions[0, :]
        self.naction = self.nactions
        self.sampled_actions_msg = Float32MultiArray()
        self.sampled_actions_msg.data = np.concatenate((np.array([0]), self.nactions.flatten())).tolist()
        self.sampled_actions_pub.publish(self.sampled_actions_msg)

        self.chosen_waypoint = self.naction[self.args.waypoint, :]
        
    def timer_callback(self):
        start = time.time()
        self.chosen_waypoint = np.zeros(4, dtype=np.float32)
        if len(self.context_queue) >= self.context_size:
            # Process observations
            self.process_images()
            
            # Use policy to get actions
            self.infer_actions()

        # Normalize and publish waypoint
        if self.naction is not None:
            if NORMALIZE:
                self.naction[:,:2] *= (self.MAX_V / self.RATE)
                self.chosen_waypoint[:2] *= (self.MAX_V / self.RATE)
            self.execute = 0 
            
            while self.execute < self.args.waypoint:
                self.waypoint = self.naction[self.execute, :]
                self.waypoint_msg.data = self.waypoint.tolist()
                self.waypoint_pub.publish(self.waypoint_msg)
                time.sleep(self.timer_period)
                self.execute += 1

        self.naction = None

        self.context_queue = []

def main(args):

    rclpy.init()
    nav_policy = NavigateVLA(args)

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
        default=4, # close waypoints exihibit straight line motion (the middle waypoint is a good default)
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
        default="",
        type=str,
        help="Prompt for the policy",
    )
    parser.add_argument(
        "--server-address",
        "-s",
        default="http://localhost:5000",
        type=str,
        help="Server address for inference",
    )
    args = parser.parse_args()
    main(args)


