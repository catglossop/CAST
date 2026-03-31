#!/usr/bin/env python3

import base64
import requests
import json
import math
import ast
import numpy as np
from numpy.lib.stride_tricks import as_strided
import sys
import csv
import copy
import time
import string
import argparse
import os
from bresenham import bresenham  # to calculate the lines connecting two grid points
from transformers import CLIPProcessor, CLIPModel

# ROS Headers
# import rospy
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_msgs.msg import Float32, Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist, PointStamped, Pose, PoseArray
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan, CompressedImage, Image
# from tf.transformations import euler_from_quaternion
from tf_transformations import euler_from_quaternion

# Headers for local costmap subscriber
from matplotlib import pyplot as plt
from matplotlib.path import Path
from PIL import Image as PILImage
from PIL import ImageChops
from openai import OpenAI

# OpenCV
import cv2

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
ORGANIZATION_ID = os.environ.get("ORGANIZATION_ID")

class Img2ground(Node):

    def __init__(self, args):

        super().__init__('ground2_img')
        self.path_prompt = "prompts/baseline.txt"
        with open(self.path_prompt, "r") as f:
            self.prompt = f.read()

        self.prompt = self.prompt.replace("INSTRUCTION", args.prompt)

        print(self.prompt)

        self.client = OpenAI(api_key=OPENAI_KEY,
                    organization=ORGANIZATION_ID)
        self.robot_radius = 0.6  # [m]
        self.x = 0.0
        self.y = 0.0
        self.v_x = 0.0
        self.v_y = 0.0
        self.w_z = 0.0
        self.goalX = 0.0006
        self.goalY = 0.0006
        self.th = 0.0

        # Params for Create 3
        self.robot_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, 
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
            )
        self.camera_height = 0.75  # height of the camera w.r.t. the robot's base/ground level
        self.camera_tilt_angle = 0
        self.camera_offset_x = 0
        self.camera_offset_y = 0  # camera y axis offset in meters
        self.P = [[393.6538, 0.0, 322.797939, 0.0], 
                  [0.0, 393.6538, 241.090902, 0.0],
                  [0.0, 0.0, 1.0, 0.0]]
        self.subOdom = self.create_subscription(Odometry, "/odom", self.odom_callback, self.robot_qos_profile)
        self.row_distance_locations = np.array([2, 3.5, 5])  # For checking occupancy
        self.point_spacing = 0.5  # in m  # For Adding Markers


        self.img_h, self.img_w = 720, 1280
        self.br = CvBridge()
        self.cv_image = None
        self.image_received = False

        # Costmap
        self.scale_percent = 300  # percent of original size
        self.costmap_shape = (200, 200)
        self.costmap_resolution = 0.05

        # self.occupancymap_low = np.zeros(self.costmap_shape, dtype=np.uint8)
        self.occupancymap_mid = np.zeros(self.costmap_shape, dtype=np.uint8)
        # self.occupancymap_high = np.zeros(self.costmap_shape, dtype=np.uint8)
        # self.occupancymap_low_inflated = np.zeros(self.costmap_shape, dtype=np.uint8)
        self.occupancymap_mid_inflated = np.zeros(self.costmap_shape, dtype=np.uint8)
        # self.occupancymap_diff_inflated = np.zeros(self.costmap_shape, dtype=np.uint8)
        self.kernel_size = (5, 5)  # for glass #(11, 11) for everything else

        self.roi_shape = (100, 100)
        self.roi_prev = np.zeros(self.roi_shape, dtype=np.uint8)
        self.roi_curr = np.zeros(self.roi_shape, dtype=np.uint8)
        self.roi_combined = np.zeros(self.roi_shape, dtype=np.uint8)
        self.diff_curr = np.zeros(self.costmap_shape, dtype=np.uint8)
        self.flag = 0
        self.counter = 0

        self.intmap_rgb = cv2.cvtColor(self.occupancymap_mid, cv2.COLOR_GRAY2RGB)
        # self.obs_low_mid_high = np.argwhere(self.occupancymap_mid > 150)  # should be null set

        # For cost map clearing
        self.height_thresh = 75  # 150
        self.occupancy_thresh = 150
        self.alpha = 0.35

        # Subscribers and Publishers
        self.subGoal = self.create_subscription(Twist, '/final_goal', self.target_callback, 10)
        self.pubRefPath = self.create_publisher(PoseArray, '/reference_path', 10)
        self.pubGoal = self.create_publisher(Twist, '/target/position', 10)
        self.subOccupancy = self.create_subscription(OccupancyGrid, "/costmap/costmap", self.occupancy_map_mid_cb, 10)
        self.subImg = self.create_subscription(Image, "front/image_raw", self.img_callback, 10)
        self.occupancy_received = False

        # VLM Parameters
        api_key = self.open_file("gamma_api.txt")  # OpenAI API Key
        print(api_key)
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

        self.marked_img = None
        self.ref_path_odom = []
        self.ref_path_img = []
        self.reprompt_thresh = 4  # in m
        self.context_prompt = self.open_file("prompts/context.txt")
        self.context_curr = None
        self.context_prev = None
        self.path_prompt = None
        self.initial = True
        self.prompted = False
        os.makedirs("images", exist_ok=True)

        # CLIP for context understanding
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Goal direction condition
        self.goal_dir_max = 1000 # degrees

        # print("GPT Script Started!")
        self.get_logger().info("GPT Script Started!")
        self.publish_goal = 1  # input("Publish final goal to planner? 1-yes, 0-no. ")

        # Timer to periodically call get_ref_path
        self.timer_period = 0.5  # seconds
        self.timer = self.create_timer(self.timer_period, self.get_ref_path)

    # Functions related to VLM
    def open_file(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as infile:
            return infile.read()

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            read = image_file.read()
            encode = base64.b64encode(read)
            decoded = encode.decode('utf-8')
        return decoded

    # Callback for Odometry
    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        rot_q = msg.pose.pose.orientation
        (roll, pitch, theta) = euler_from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w])
        self.th = theta
        # print("Theta of body wrt odom:", self.th / 0.0174533)

        # Get robot's current velocities
        self.v_x = msg.twist.twist.linear.x
        self.v_y = msg.twist.twist.linear.y
        self.w_z = msg.twist.twist.angular.z
        # print("Robot's current velocities", [self.v_x, self.w_z])

    def target_callback(self, data):
        self.get_logger().info("---------------Inside Goal Callback------------------------")

        radius = data.linear.x  # this will be r
        theta = data.linear.y * 0.0174533  # this will be theta
        self.get_logger().info(f"r and theta: {data.linear.x}, {data.linear.y}")

        # Goal wrt robot frame
        goalX_rob = radius * math.cos(theta)
        goalY_rob = radius * math.sin(theta)

        # Goal wrt odom frame (from where robot started)
        self.goalX = self.x + goalX_rob * math.cos(self.th) - goalY_rob * math.sin(self.th)
        self.goalY = self.y + goalX_rob * math.sin(self.th) + goalY_rob * math.cos(self.th)

        # print("Self odom:", self.x, self.y)
        # print("Goals wrt odom frame:", self.goalX, self.goalY)

        # If goal is published as x, y coordinates wrt odom uncomment this
        # self.goalX = data.linear.x
        # self.goalY = data.linear.y

    def img_callback(self, msg):
        self.cv_image = self.br.imgmsg_to_cv2(msg, "bgr8")  # for images
        cv2.imwrite("images/img.png", self.cv_image)
        self.img_h, self.img_w, _ = np.shape(self.cv_image)
        self.image_received = True

    def occupancy_map_mid_cb(self, msg):
        self.occupancymap_mid_inflated = self.process_costmap(msg)
        self.occupancy_received = True

    # Process the occupancy map
    def process_costmap(self, data):
        # Mid occupancy Map
        int_mid = np.reshape(data.data, (-1, int(math.sqrt(len(data.data)))))
        int_mid = np.reshape(data.data, (int(math.sqrt(len(data.data))), -1))
        int_mid = np.flip(int_mid, axis=0)
        # int_mid = np.rot90(np.fliplr(int_mid), 1, (1, 0)) + 128  # Shift the range from [-128,128] to [0,256]

        im_mid_image = PILImage.fromarray(np.uint8(int_mid))
        yaw_deg = -90  # Now cost map published wrt baselink
        im_mid_pil = im_mid_image.rotate(-yaw_deg)
        self.occupancymap_mid = np.array(im_mid_pil)
        # self.occupancymap_mid = np.rot90(np.uint8(self.occupancymap_mid), 2)  # Rotating by 180 deg

        # Inflation
        kernel = np.ones(self.kernel_size, np.uint8)
        dilated_mid = cv2.dilate(self.occupancymap_mid, kernel, iterations=1)
        self.occupancymap_mid_inflated = np.array(dilated_mid)

        return np.array(dilated_mid)

    # Convert odom coordinates to robot frame
    def odom_to_robot(self, x_odom, y_odom):
        x_rob_odom_list = np.asarray([round(self.x, 2) for _ in range(x_odom.shape[0])])
        y_rob_odom_list = np.asarray([round(self.y, 2) for _ in range(y_odom.shape[0])])

        x_rob = (x_odom - x_rob_odom_list) * math.cos(self.th) + (y_odom - y_rob_odom_list) * math.sin(self.th)
        y_rob = -(x_odom - x_rob_odom_list) * math.sin(self.th) + (y_odom - y_rob_odom_list) * math.cos(self.th)
        # print("Trajectory end-points wrt robot:", x_rob, y_rob)

        return x_rob, y_rob

    # Convert robot frame coordinates to costmap coordinates
    def robot_to_costmap(self, x_rob, y_rob):
        costmap_shape_list_0 = [self.costmap_shape[0] / 2 for _ in range(y_rob.shape[0])]
        costmap_shape_list_1 = [self.costmap_shape[1] / 2 for _ in range(x_rob.shape[0])]

        y_list = [int(y / self.costmap_resolution) for y in y_rob]
        x_list = [int(x / self.costmap_resolution) for x in x_rob]

        cm_col = np.asarray(costmap_shape_list_0) - np.asarray(y_list)
        cm_row = np.asarray(costmap_shape_list_1) - np.asarray(x_list)
        # print("Costmap coordinates of end-points: ", (int(cm_row), int(cm_col)))

        return cm_col.astype(int), cm_row.astype(int)

    # Convert robot frame coordinates to odom frame
    def robot_to_odom(self, x_rob, y_rob):
        self_x_list = [self.x for _ in range(x_rob.shape[0])]
        self_y_list = [self.y for _ in range(y_rob.shape[0])]

        x_odom = self_x_list + x_rob * math.cos(self.th) - y_rob * math.sin(self.th)
        y_odom = self_y_list + x_rob * math.sin(self.th) + y_rob * math.cos(self.th)

        return x_odom, y_odom

    # Check occupancy of ground locations
    def check_occupancy(self):
        x_list = []
        y_list = []

        col_rob = int(self.costmap_shape[0] / 2)
        row_rob = int(self.costmap_shape[0] / 2)

        step_size = int(self.point_spacing / self.costmap_resolution)

        start = int(0.5 / self.costmap_resolution)
        init_col = int((6 * start) - 1)  # Based on costmap size and resolution
        end_col = int(self.costmap_shape[0]) - (6 * start)

        dist_list = self.row_distance_locations
        row_rob_list = [row_rob for _ in range(dist_list.shape[0])]
        cm_row_list = np.asarray(row_rob_list) - (np.asarray(dist_list) / self.costmap_resolution)

        for dist_row in cm_row_list:
            for l in range(init_col, end_col, step_size):
                line_cells = np.array(list(bresenham(int(dist_row), l, row_rob, col_rob)))
                line_values = self.occupancymap_mid_inflated[line_cells[:, 0], line_cells[:, 1]]
                if max(line_values) < self.occupancy_thresh:
                    y_wrt_rob = (col_rob - l) * self.costmap_resolution
                    x_wrt_rob = (row_rob - dist_row) * self.costmap_resolution
                    x_list.append(x_wrt_rob)
                    y_list.append(y_wrt_rob)
                else:
                    # print("Occupied candidate waypoints :", (l * self.costmap_resolution, dist_row * self.costmap_resolution))
                    pass

        cm_col, cm_row = self.robot_to_costmap(np.array(x_list), np.array(y_list))
        x_list_odom, y_list_odom = self.robot_to_odom(np.array(x_list), np.array(y_list))  # Coordinates w.r.t. odom frame
        odom_coord_list = np.column_stack((x_list_odom, y_list_odom))
        map_coord_list = np.column_stack((cm_col, cm_row))
        occ_map = cv2.cvtColor(self.occupancymap_mid_inflated, cv2.COLOR_GRAY2RGB)
        occ_map = cv2.circle(occ_map, (int(self.costmap_shape[0]/2), int(self.costmap_shape[0]/2)), radius=5, color=(255, 255,0), thickness=-1)
        for l in range(len(x_list)):
            occ_map = cv2.circle(occ_map, (cm_col[l], cm_row[l]), radius=5, color=(0, 255, 255), thickness=-1)

        cv2.imwrite("images/map.png", cv2.resize(occ_map, (600,600), interpolation = cv2.INTER_AREA))

        return x_list, y_list, odom_coord_list, map_coord_list

    # Add visual markers to the image
    def add_visual_markers(self):
        marked_img = np.zeros((self.img_h, self.img_w))

        obs_free_xyz = []
        x_list, y_list, odom_coord_list, map_coord_list = self.check_occupancy()

        for j in range(len(x_list)):
            obs_free_xyz.append([-y_list[j] + self.camera_offset_y, self.camera_height, x_list[j] - self.camera_offset_x])

        points = np.array(obs_free_xyz)
        if len(points) > 0:
            # Transform x,y,z ground coordinates to camera frame
            alpha = np.deg2rad(self.camera_tilt_angle)
            Rotation_mat = [[1, 0, 0],
                            [0, np.cos(alpha), -np.sin(alpha)],
                            [0, np.sin(alpha), np.cos(alpha)]]

            points_rotated = np.matmul(points, Rotation_mat)

            # Reshape points
            X0 = np.ones((points_rotated.shape[0], 1))
            pointsnew = np.hstack((points_rotated, X0))

            # Projecting ground locations to image plane
            uvw = np.dot(self.P, np.transpose(pointsnew))

            u_vec = uvw[0]
            v_vec = uvw[1]
            w_vec = uvw[2]

            x_vec = u_vec / w_vec
            y_vec = v_vec / w_vec

            # Remove points located out of the RGB image
            new_xvec = []
            new_yvec = []

            for i in range(len(y_vec)):
                if (0 <= y_vec[i] <= self.img_h) and (0 <= x_vec[i] <= self.img_w):
                    new_xvec.append([int(x_vec[i]), odom_coord_list[i][0], odom_coord_list[i][1],
                                     map_coord_list[i][0], map_coord_list[i][1]])
                    new_yvec.append(int(y_vec[i]))

            # Check the unique number of rows in the coordinates set to identify number of waypoint rows exists
            seen = set()
            uniq = sorted(([x for x in np.array(new_yvec) if x not in seen and not seen.add(x)]))

            final_x_img = []
            final_y_img = []
            row_id = []
            final_x_odom = []
            final_y_odom = []
            final_col_map = []
            final_row_map = []

            row_no = 1
            for k in uniq:
                indices = [m for m in range(len(new_yvec)) if new_yvec[m] == k]
                row_coordx = sorted([new_xvec[l] for l in indices])
                row_coordy = [new_yvec[p] for p in indices]
                row_id.extend(np.ones(len(row_coordx)) * row_no)
                final_x_img.extend(np.array(row_coordx)[:, 0].astype(int))
                final_y_img.extend(row_coordy)
                final_x_odom.extend(np.array(row_coordx)[:, 1])
                final_y_odom.extend(np.array(row_coordx)[:, 2])
                final_col_map.extend(np.array(row_coordx)[:, 3])
                final_row_map.extend(np.array(row_coordx)[:, 4])
                row_no += 1

            final_image_coords = np.array(self.merge(final_x_img, final_y_img))  # Obs free image locations

            # Display image project points
            if self.cv_image is not None:
                marked_img = self.cv_image.copy()
                red = (0, 0, 255)
                counter = 1
                img_markers = []  # The list of text numbers drawing on top of the obs free image coordinates
                for l in range(len(final_image_coords)):
                    if row_id[l] == 1:
                        marked_img = cv2.putText(marked_img, str(counter),
                                                 (final_image_coords[l][0], final_image_coords[l][1]),
                                                 cv2.FONT_HERSHEY_SIMPLEX, 1.2, red, 4, cv2.LINE_AA)
                        img_markers.append(counter)
                        counter += 1
                    elif row_id[l] == 2:
                        marked_img = cv2.putText(marked_img, str(counter),
                                                 (final_image_coords[l][0], final_image_coords[l][1]),
                                                 cv2.FONT_HERSHEY_SIMPLEX, 1.2, red, 4, cv2.LINE_AA)
                        img_markers.append(counter)
                        counter += 1
                    elif row_id[l] == 3:
                        marked_img = cv2.putText(marked_img, str(counter),
                                                 (final_image_coords[l][0], final_image_coords[l][1]),
                                                 cv2.FONT_HERSHEY_SIMPLEX, 1.2, red, 4, cv2.LINE_AA)
                        img_markers.append(counter)
                        counter += 1

                img_markers = np.asarray(img_markers)  # Array of marker IDs corresponding to each point

            final_odom_coords = np.column_stack((final_x_odom, final_y_odom))
            final_map_coords = np.column_stack((final_col_map, final_row_map))

        else:
            if self.cv_image is not None:
                marked_img = self.cv_image.copy()
            else:
                marked_img = np.zeros((self.img_h, self.img_w))

        return marked_img, row_id, img_markers, final_image_coords, final_odom_coords, final_map_coords

    # Compare two strings
    def compare_strings(self, a, b):
        return [c for c in a if c.isalpha()] == [c for c in b if c.isalpha()]

    # Choose the appropriate prompt based on context
    # def choose_prompt(self, context):
    #     if self.compare_strings(context.lower(), 'indoor corridor'):
    #         self.get_logger().info("Context is indoor corridor.")
    #         self.path_prompt = self.open_file("prompts/corridor.txt")

    #     elif self.compare_strings(context.lower(), 'outdoor terrain'):
    #         self.get_logger().info("Context is outdoor terrain.")
    #         self.path_prompt = self.open_file("prompts/outdoor_terrain.txt")

    #     elif self.compare_strings(context.lower(), 'crosswalk'):
    #         self.get_logger().info("Context is crosswalk.")
    #         self.path_prompt = self.open_file("prompts/crosswalk.txt")

    #     elif self.compare_strings(context.lower(), 'scenario with people'):
    #         self.get_logger().info("Context is scenario with interacting people.")
    #         self.path_prompt = self.open_file("prompts/social1.txt")

    #     return self.path_prompt

    # Use CLIP to determine context
    def ask_clip(self, image):
        # 0 - Corridor, 1 - outdoor terrain, 2 - crosswalk, 3 - scenario with people
        inputs = self.clip_processor(text=["a photo of an indoor corridor", "a photo of outdoor terrains",
                                           "a photo of a zebra crossing", "a photo of people in a corridor"],
                                     images=image, return_tensors="pt", padding=True)
        outputs = self.clip_model(**inputs)
        logits_per_image = outputs.logits_per_image  # Image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # Label probabilities
        c_max = np.argmax(probs.detach().numpy())

        if c_max == 0:
            return "indoor corridor"
        elif c_max == 1:
            return "outdoor terrain"
        elif c_max == 2:
            return "crosswalk"
        elif c_max == 3:
            return "scenario with people"

    # Ask GPT for reference path
    def ask_gpt(self, base64_image, prompt, context_checking):
        message = [{"role": "user",
                   "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", 
                         "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
                ],
                },
                ]
        t1 = time.time()
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=message,
            max_tokens=1000,
        )
        t2 = time.time()
        self.get_logger().info(f"Inference time: {t2 - t1}")

        content = response.choices[0].message.content

        if context_checking:
            # Print the content
            self.get_logger().info("VLM: Checking the current context.")
            return content

        else:
            # Obtain scores/points from GPT's response and print them
            self.get_logger().info("VLM: Getting new points for reference path.")
            points_list = ast.literal_eval(content)
            self.get_logger().info(f"Points from VLM: {points_list}")
            return points_list


    # Construct the reference path
    def construct_ref_path_2(self, marked_img, points_list, row_id, img_markers, final_image_coords, final_odom_coords, final_map_coords):
        num_rows = int(max(row_id))
        ref_path_img = []
        ref_path_odom = []
        ref_path_costmap = []

        if isinstance(points_list, list):
            final_list = points_list
        else:
            final_list = list(points_list)
        st = set(final_list)
        idx = [i for i, e in enumerate(img_markers) if e in st]

        for i in idx:
            ref_path_img.append([final_image_coords[i][0], final_image_coords[i][1]])
            ref_path_odom.append([final_odom_coords[i][0], final_odom_coords[i][1]])
            ref_path_costmap.append([final_map_coords[i][0], final_map_coords[i][1]])
        # Publish reference path wrt odom
        ref_path = PoseArray()
        for i in range(len(idx)):
            goal = Pose()
            goal.position.x = ref_path_odom[i][0]
            goal.position.y = ref_path_odom[i][1]
            goal.orientation.w = 1.0
            ref_path.poses.append(goal)
        self.pubRefPath.publish(ref_path)

        return ref_path_odom, ref_path_img

    # Get the reference path
    def get_ref_path(self):
        if not self.image_received or not self.occupancy_received:
            if not self.image_received:
                print("waiting for image")
            if not self.occupancy_received:
                print("waiting for map")
            return

        # Add markers to image
        self.marked_img, row_id, img_markers, final_image_coords, final_odom_coords, final_map_coords = self.add_visual_markers()

        # Resizing the marked image
        marked_img_resized = cv2.resize(self.marked_img, (0, 0), fx=0.9, fy=0.9, interpolation=cv2.INTER_AREA)
        img_h, img_w, _ = np.shape(marked_img_resized)
        self.get_logger().info(f"Image size: {img_h}, {img_w}")

        # Encode the marked image
        image_path =  'images/1.jpg'
        cv2.imwrite(image_path, marked_img_resized)
        base64_image = self.encode_image(image_path)

        x_goal_rob, y_goal_rob = self.odom_to_robot(np.array([self.goalX]), np.array([self.goalY]))

        goal_direction = 0
        if abs(x_goal_rob) > 0.5 and self.goalX != 0.0006:
            goal_direction = math.degrees(math.atan2(y_goal_rob, x_goal_rob))
            self.get_logger().info(f"Goal direction: {goal_direction}")
        else:
            if self.goalY != 0.0006 and y_goal_rob > 0:  # left
                goal_direction = 90  # in degrees
                self.get_logger().info("Goal is close to being perpendicularly to the left.")

            elif self.goalY != 0.0006 and y_goal_rob < 0:  # right
                goal_direction = -90  # in degrees
                self.get_logger().info("Goal is close to being perpendicularly to the right.")

        if abs(goal_direction) <= self.goal_dir_max:
            # Check context and query VLM for reference path
            if self.initial:
                self.initial = False
                points_list = self.ask_gpt(base64_image, self.prompt, False)
                print("Initial gpt points list: ", points_list)
                self.ref_path_odom, self.ref_path_img = self.construct_ref_path_2(self.marked_img, points_list, row_id,
                                                                                 img_markers, final_image_coords,
                                                                                 final_odom_coords, final_map_coords)
                print("Ref path: ", self.ref_path_odom)
                self.goalX = 10.0
                self.goalY = 0.0
                self.context_prev = self.context_curr
                self.prompted = True

            elif not self.initial and len(self.ref_path_odom) > 0:

                dist_to_ref_point = np.linalg.norm(np.array([self.x, self.y]) - np.array(
                    [self.ref_path_odom[0][0], self.ref_path_odom[0][1]]))
                self.get_logger().info(f"Dist from current ref path point: {dist_to_ref_point}")
                if self.prompted and dist_to_ref_point < self.reprompt_thresh:
                    self.get_logger().info("Time to prompt!")
                    self.prompted = False

                if not self.prompted and dist_to_ref_point < self.reprompt_thresh:
                    self.get_logger().info("Context remains the same")
                    points_list = self.ask_gpt(base64_image, self.prompt, False)
                    self.ref_path_odom, self.ref_path_img = self.construct_ref_path_2(self.marked_img, points_list,
                                                                                        row_id, img_markers,
                                                                                        final_image_coords,
                                                                                        final_odom_coords,
                                                                                        final_map_coords)
                    self.context_prev = self.context_curr
                    self.prompted = True

                    self.get_logger().info("Got new reference path!")
                    self.get_logger().info(f"Dist from current ref path point: {dist_to_ref_point}")


            elif not self.initial and len(self.ref_path_odom) == 0:
                self.get_logger().info("No reference path obtained.")

            # Display image
            if len(self.ref_path_img) >= 2:
                for i in range(len(self.ref_path_img) - 1):
                    cv2.line(self.marked_img, tuple(self.ref_path_img[i]), tuple(self.ref_path_img[i + 1]), (0, 255, 0), 3)

            cv2.imwrite("images/img_with_points.png", self.marked_img)

        else:
            self.get_logger().info("Goal outside FOV! No reference path published.")
        print(f"GOAL: {self.goalX} {self.goalY}")
        # Publish final goal to robot
        if self.publish_goal == 1 and (self.goalX != 0.0006) and (self.goalY != 0.0006):
            print("publishing goal")
            final_goal = Twist()
            final_goal.linear.x = self.goalX
            final_goal.linear.y = self.goalY
            self.pubGoal.publish(final_goal)
        else:
            self.get_logger().info("Not publishing final goal yet.")

    # Merge two lists into a list of tuples
    def merge(self, x_vec, y_vec):
        merged_list = [(int(x_vec[i]), int(y_vec[i])) for i in range(len(x_vec))]
        return merged_list


def main(args):
    # rospy.init_node('ground2_img')
    rclpy.init()
    img2ground = Img2ground(args)
    # while not rospy.is_shutdown():
    #     img2ground.get_ref_path()
    try:
        rclpy.spin(img2ground)
    except KeyboardInterrupt:
        pass
    finally:
        img2ground.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default="Move to the chair")
    args = parser.parse_args()
    main(args)
