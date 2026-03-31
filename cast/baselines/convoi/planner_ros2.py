#!/usr/bin/env python3

# Author: Connor McGuile
# Latest author: Adarsh Jagan Sathyamoorthy

# A custom Dynamic Window Approach implementation for use with Turtlebot.
# Obstacles are registered by a front-mounted laser and stored in a set.
# If, for testing purposes or otherwise, you do not want the laser to be used,
# disable the laserscan subscriber and create your own obstacle set in main(),
# before beginning the loop. If you do not want obstacles, create an empty set.
# Implementation based off Fox et al.'s paper, The Dynamic Window Approach to
# Collision Avoidance (1997).

# VERSION NOTES 
# This version detects regions of transparent obstacles and linearly extrapolates them.
# It uses a costmap for detecting obstacles instead of raw scans. 
# NOTE: Obstacle costs for the trajectories are calculated based on distance to obstacle grids in the costmap and not by
# checking if the trajectory lies on the obstacle. The latter is inefficient and could lead to collisions in some cases. 

# TODO: Add subscriber callback function for reference image from VLM (GPT-4v).
# TODO: Integrate function that can handle vegetation using MIMs (Multi-layer Intensity Maps).
# TODO: Make the code robot independent. 

import rclpy
from rclpy.node import Node
import math
import numpy as np
from numpy.lib.stride_tricks import as_strided
from std_msgs.msg import Float32, Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError

from geometry_msgs.msg import Twist, PoseArray, PointStamped
from nav_msgs.msg import OccupancyGrid, Odometry
import sensor_msgs.msg
from sensor_msgs.msg import LaserScan, CompressedImage
from tf_transformations import euler_from_quaternion
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import time
import sys
import csv

# Headers for local costmap subscriber
from matplotlib import pyplot as plt
from matplotlib.path import Path
from PIL import Image

import sys
import copy
import cv2


class Config():
    # simulation parameters

    def __init__(self):
        
        # Robot parameters
        self.max_speed = 0.25     # [m/s]
        self.min_speed = 0.0      # [m/s]
        self.max_yawrate = 0.4    # [rad/s]
        self.max_accel = 1        # [m/ss]
        self.max_dyawrate = 3.2   # [rad/ss]
        
        self.v_reso = 0.10        # [m/s]
        self.yawrate_reso = 0.2   # [rad/s]
        
        self.dt = 0.5             # [s]
        self.predict_time = 2.0   # [s]
        
        # Cost gains
        self.to_goal_cost_gain = 10.0         
        self.obs_cost_gain = 10.0             
        self.speed_cost_gain = 0.1            
        self.ref_path_cost_gain = 7.5         
        
        self.robot_radius = 0.65  # [m]
        self.x = 0.0
        self.y = 0.0
        self.v_x = 0.0
        self.v_y = 0.0
        self.w_z = 0.0
        self.goalX = 0.0006
        self.goalY = 0.0006
        self.th = 0.0

        self.collision_threshold = 0.3 # [m]
        self.conf_thresh = 0.80

        # DWA output
        self.min_u = []

        self.stuck_status = False
        self.stuck_count = 0
        self.pursuing_safe_loc = False
        self.okay_locations = []
        self.stuck_locations = []

        # Costmap
        self.scale_percent = 300 # percent of original size
        self.costmap_shape = (200, 200)
        self.costmap_resolution = 0.05
        print("MIM Started!")
        
        self.intensitymap_mid = np.zeros(self.costmap_shape, dtype=np.uint8)
        self.intensitymap_mid_inflated = np.zeros(self.costmap_shape, dtype=np.uint8)
        self.glass_costmap = np.zeros(self.costmap_shape, dtype=np.uint8)
        self.glass_costmap_inflated = np.zeros(self.costmap_shape, dtype=np.uint8)
        self.planning_costmap = np.zeros(self.costmap_shape, dtype=np.uint8)
        self.viz_costmap = cv2.cvtColor(self.glass_costmap, cv2.COLOR_GRAY2RGB)
        self.viz_costmap_inflated = cv2.cvtColor(self.glass_costmap_inflated, cv2.COLOR_GRAY2RGB)
        self.kernel_size = (9, 9) # for glass #(11, 11) for everything else
        
        self.roi_shape = (100, 100)
        self.roi_prev = np.zeros(self.roi_shape, dtype=np.uint8)
        self.roi_curr = np.zeros(self.roi_shape, dtype=np.uint8)
        self.roi_combined = np.zeros(self.roi_shape, dtype=np.uint8)
        self.diff_curr = np.zeros(self.costmap_shape, dtype=np.uint8)
        self.flag = 0
        self.counter = 0

        self.obs_consideration_thresh = 100

        # For cost map clearing
        self.height_thresh = 75 #150
        self.intensity_thresh = 180
        self.alpha = 0.35

        # Glass shadow isolation 
        self.glass_intensity_upper_th = 150 # threshold intensities that reflect the range of the glass shadow in the mid map
        self.glass_intensity_lower_th = 90
        self.noise_removal_threshold = 15 # blob size threshold to remove small noise patches
        self.intmap_mid_binary_thresh = 10 # threshold to convert grayscale mid intensity map to a binary image

        self.clearing_time = 25 #50

        # Linear Extrapolation
        self.line_length = 20
        self.line_thickness = 1

        # Map inflation
        self.uniform_inflation_kernel = np.ones(self.kernel_size, np.uint8)

        # VLM Reference path
        self.ref_path = []
        self.x_ref_odom = []
        self.y_ref_odom = []
        self.waypoint_curr = None
        self.x_ref_max = 0.0
        self.y_ref_avg = 0.0

        # Goal direction condition
        self.goal_dir_max = 75 # degrees
        self.wp_dir_max = 60

        self.head_to_goal_thresh = 2

    # Callback for Odometry
    def assignOdomCoords(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        rot_q = msg.pose.pose.orientation
        (roll,pitch,theta) = euler_from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w])
        self.th = theta

        # Get robot's current velocities
        self.v_x = msg.twist.twist.linear.x
        self.v_y = msg.twist.twist.linear.y
        self.w_z = msg.twist.twist.angular.z 

    # Callback for goal
    def target_callback(self, data):
        print("---------------Inside Final Goal Callback------------------------")
        self.goalX = data.linear.x
        self.goalY = data.linear.y

    def ref_path_callback(self, data):
        self.ref_path = []
        self.x_ref_odom = []
        self.y_ref_odom = []
        if len(data.poses) > 0:
            for i in range(len(data.poses)):
                print("Reference waypoint ", i, " : ", data.poses[i].position.x, data.poses[i].position.y)
                self.ref_path.append((data.poses[i].position.x, data.poses[i].position.y))
                self.x_ref_odom.append(data.poses[i].position.x)
                self.y_ref_odom.append(data.poses[i].position.y)

            self.waypoint_curr = self.ref_path[len(data.poses)-1] # Choose the closest point as first waypoint
            print("WAYPOINT: ", self.waypoint_curr)
        else:
            self.ref_path = []
            self.x_ref_odom = []
            self.y_ref_odom = []
            self.waypoint_curr = None

    # Callback for MID intensity map
    def intensity_map_mid_cb(self, data):
        print("Got occupancy grid")
        # Mid Intensity Map
        int_mid = np.reshape(data.data, (-1, int(math.sqrt(len(data.data)))))
        int_mid = np.reshape(data.data, (int(math.sqrt(len(data.data))), -1))
        int_mid = np.flip(int_mid, axis=0)

        im_mid_image = Image.fromarray(np.uint8(int_mid))
        yaw_deg = -90 # Now cost map published wrt baselink
        im_mid_pil = im_mid_image.rotate(-yaw_deg)
        self.intensitymap_mid = np.array(im_mid_pil)

        self.glass_detect()

    def glass_detect(self):
        # Center of costmap
        center_x = int(self.costmap_shape[0]/2)
        center_y = int(self.costmap_shape[1]/2)
        roi_center_x = int(self.roi_shape[0]/2)
        roi_center_y = int(self.roi_shape[1]/2)
        roi_side = int(self.roi_shape[0]/2)        # roi_side is used to set the range from a center for the roi (center - roi_side, center + roi_side)
        roi_l = int(self.roi_shape[0])

        ## Thresholding the mid intensity map to isolate glass shadows
        intmap_mid_thresholded = copy.copy(self.intensitymap_mid)

        # Thresholding all the objects out of the intensity range of the glass shadow to 0 in the mid map
        # intmap_mid_thresholded will only have white patches in regions falling within intensity thresholds
        intmap_mid_thresholded[intmap_mid_thresholded > self.glass_intensity_upper_th] = 0
        intmap_mid_thresholded[intmap_mid_thresholded < self.glass_intensity_lower_th] = 0

        # Considering the ROI of the thresholded mid map and the raw low map
        intmap_mid_thresholded_roi = intmap_mid_thresholded[int(center_x-roi_side):int(center_x+roi_side), int(center_y-roi_side):int(center_y+roi_side)]

        # Dilate the ROIs
        intmap_mid_thresholded_roi = cv2.dilate(intmap_mid_thresholded_roi, self.uniform_inflation_kernel, iterations=1)
        
        # Convert the isolated glass shadow ROI to a binary image        
        _, glass_shadow_roi_bw = cv2.threshold(intmap_mid_thresholded_roi, self.intmap_mid_binary_thresh, 255, cv2.THRESH_BINARY)

        # Identify contours in the isolated glass shadow ROI to remove noisy contours
        glass_shadow_cleaned = np.zeros_like(glass_shadow_roi_bw)
        glass_shadow_rgb = cv2.cvtColor(glass_shadow_cleaned, cv2.COLOR_GRAY2RGB)
        line_pt1_roi = np.array([0, 0])
        line_pt2_roi = np.array([0, 0])

        line_pt1_list = []
        line_pt2_list = []

        contours, _ = cv2.findContours(glass_shadow_roi_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.noise_removal_threshold:
                M = cv2.moments(contour)
                # calculate x,y coordinate of center
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Linear Extrapolation for planning (wrt ROI)
                norm = math.sqrt((roi_center_x - cX)**2 + (roi_center_y - cY)**2)
                line_pt1_roi = np.array([cX, cY]) + self.line_length * np.array([(cY - roi_center_y)/norm, -(cX - roi_center_x)/norm])
                line_pt2_roi = np.array([cX, cY]) - self.line_length * np.array([(cY - roi_center_y)/norm, -(cX - roi_center_x)/norm])
                delta_x_1 = (line_pt1_roi[0] - roi_center_x)
                delta_y_1 = (line_pt1_roi[1] - roi_center_y)
                delta_x_2 = (line_pt2_roi[0] - roi_center_x)
                delta_y_2 = (line_pt2_roi[1] - roi_center_y)
                line_pt1_cm = np.array([center_x, center_y]) + np.array([delta_x_1, delta_y_1])
                line_pt2_cm = np.array([center_x, center_y]) + np.array([delta_x_2, delta_y_2])

                line_pt1_list.append(line_pt1_cm)
                line_pt2_list.append(line_pt2_cm)

                # Planning costmap: Add ROI with noise-removed glass shadow and extrapolated line segment to low map
                glass_shadow_cleaned = cv2.line(glass_shadow_cleaned, (int(line_pt1_roi[0]), int(line_pt1_roi[1])), (int(line_pt2_roi[0]), int(line_pt2_roi[1])), 255, thickness=self.line_thickness)

        # Update planning costmap
        # CHANGED TO MID
        self.intensitymap_mid[int(center_x-roi_side):int(center_x+roi_side), int(center_y-roi_side):int(center_y+roi_side)] = glass_shadow_cleaned + self.intensitymap_mid[int(center_x-roi_side):int(center_x+roi_side), int(center_y-roi_side):int(center_y+roi_side)]
        self.glass_costmap = self.intensitymap_mid

        # Visualization costmap
        self.viz_costmap = cv2.cvtColor(self.glass_costmap, cv2.COLOR_GRAY2RGB)
        cv2.circle(self.viz_costmap, (center_x, center_y), 2, (255, 0, 255), -1)  # Robot position

        # Draw lines
        for i in range(len(line_pt1_list)):
            cv2.line(self.viz_costmap, (int(line_pt1_list[i][0]), int(line_pt1_list[i][1])), (int(line_pt2_list[i][0]), int(line_pt2_list[i][1])), (0, 0, 255), thickness=self.line_thickness)

        # Inflate the planning and visualization costmaps
        self.glass_costmap_inflated = cv2.dilate(self.glass_costmap, self.uniform_inflation_kernel, iterations=1)
        self.viz_costmap_inflated = cv2.dilate(self.viz_costmap, self.uniform_inflation_kernel, iterations=1)


class Obstacles():
    def __init__(self):
        # Set of coordinates of obstacles in view
        self.obst = set()
        self.collision_status = False

    # Custom range implementation to loop over LaserScan degrees with
    # a step and include the final degree
    def myRange(self, start, end, step):
        i = start
        while i < end:
            yield i
            i += step
        yield end

    # Callback for LaserScan
    def assignObs(self, msg, config):
        deg = len(msg.ranges)   # Number of degrees - varies in Sim vs real world
        self.obst = set()       # reset the obstacle set to only keep visible objects

        maxAngle = 360
        scanSkip = 1
        anglePerSlot = (float(maxAngle) / deg) * scanSkip
        angleCount = 0
        angleValuePos = 0
        angleValueNeg = 0
        self.collision_status = False
        for angle in self.myRange(0, deg-1, scanSkip):
            distance = msg.ranges[angle]

            if (distance < 0.05) and (not self.collision_status):
                self.collision_status = True
                pass  # Replace with appropriate collision handling

            if angleCount < (deg / (2*scanSkip)):
                angleValueNeg += anglePerSlot
                scanTheta = (angleValueNeg - 180) * math.pi/180.0
            elif angleCount > (deg / (2*scanSkip)):
                angleValuePos += anglePerSlot
                scanTheta = angleValuePos * math.pi/180.0
            else:
                scanTheta = 0

            angleCount += 1

            if distance < 4:
                objTheta =  scanTheta + config.th
                obsX = round((config.x + (distance * math.cos(abs(objTheta))))*8)/8
                if objTheta < 0:
                    obsY = round((config.y - (distance * math.sin(abs(objTheta))))*8)/8
                else:
                    obsY = round((config.y + (distance * math.sin(abs(objTheta))))*8)/8

                # add coords to set so as to only take unique obstacles
                self.obst.add((obsX, obsY))


# Motion model
def motion(x, u, dt):
    # x = [x(m), y(m), theta(rad), v(m/s), omega(rad/s)]
    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt

    x[3] = u[0]
    x[4] = u[1]

    return x

# Calculate dynamic window
def calc_dynamic_window(x, config):
    Vs = [config.min_speed, config.max_speed,
          -config.max_yawrate, config.max_yawrate]

    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_dyawrate * config.dt,
          x[4] + config.max_dyawrate * config.dt]

    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    return dw

# Calculate trajectory
def calc_trajectory(xinit, v, y, config):
    x = np.array(xinit)
    traj = np.array(x)  # many motion models stored per trajectory
    time = 0
    while time <= config.predict_time:
        x = motion(x, [v, y], config.dt)
        traj = np.vstack((traj, x))
        time += config.dt  # next sample
    return traj

# Calculate final input
def calc_final_input(x, u, dw, config, ob):
    xinit = x[:]
    min_cost = float("inf")
    config.min_u = u
    config.min_u[0] = 0.0
    
    yellow = (0, 255, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    orange = (0, 150, 255)

    count = 0

    config.glass_costmap_inflated = cv2.dilate(config.glass_costmap, config.uniform_inflation_kernel, iterations=1)
    config.viz_costmap_inflated = cv2.dilate(config.viz_costmap, config.uniform_inflation_kernel, iterations=1)

    # evaluate all trajectory with sampled input in dynamic window
    for v in np.arange(dw[0], dw[1] + config.v_reso/2, config.v_reso):
        for w in np.arange(dw[2], dw[3] + config.yawrate_reso/2, config.yawrate_reso):
            count += 1 
            traj = calc_trajectory(xinit, v, w, config)

            # calc costs with weighted gains
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(traj, config)
            speed_cost = config.speed_cost_gain * (config.max_speed - traj[-1, 3]) # end v should be as close to max_speed to have low cost

            traj_cm_col, traj_cm_row = transform_traj_to_costmap(config, traj)
            obs_cost = config.obs_cost_gain * calc_obs_cost(config, traj, traj_cm_col, traj_cm_row)

            if (config.waypoint_curr is not None) and goal_within_fov(config) and \
                np.linalg.norm(np.array([config.x, config.y]) - np.array([config.goalX, config.goalY])) > config.head_to_goal_thresh:
                ref_path_cost = config.ref_path_cost_gain * calc_ref_path_follow_cost_2(config, traj)
            else:
                ref_path_cost = 0
                print("Not following ref path. Heading to final goal!")

            final_cost = to_goal_cost + obs_cost + speed_cost + ref_path_cost

            config.viz_costmap_inflated = draw_traj(config, traj, yellow)

            # search minimum trajectory
            if min_cost >= final_cost:
                min_cost = final_cost
                config.min_u = [v, w]

    traj = calc_trajectory(xinit, config.min_u[0], config.min_u[1], config)
    to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(traj, config)
    traj_cm_col, traj_cm_row = transform_traj_to_costmap(config, traj)
    obs_cost_min = config.obs_cost_gain * calc_obs_cost(config, traj, traj_cm_col, traj_cm_row)
    
    if (config.waypoint_curr is not None) and goal_within_fov(config) and \
        np.linalg.norm(np.array([config.x, config.y]) - np.array([config.goalX, config.goalY])) > config.head_to_goal_thresh:
        ref_path_cost = config.ref_path_cost_gain * calc_ref_path_follow_cost_2(config, traj)
    else:
        ref_path_cost = 0
        print("Not following ref path. Heading to final goal!")
    
    config.viz_costmap_inflated = draw_traj(config, traj, green)
    return config.min_u

def goal_within_fov(config):
    x_goal_rob, y_goal_rob = odom_to_robot(config, np.array([config.goalX]), np.array([config.goalY]))
    if abs(x_goal_rob) > 0.5: 
        goal_direction = math.degrees(math.atan(y_goal_rob / x_goal_rob)) # in deg
    else:
        if y_goal_rob > 0:  # left
            goal_direction = 90
        elif y_goal_rob < 0:  # right
            goal_direction = -90
    if abs(goal_direction) <= config.goal_dir_max:
        return True
    else:
        return False

def crossed_current_waypoint(config, x):
    if math.sqrt((x[0] - config.waypoint_curr[0])**2 + (x[1] - config.waypoint_curr[1])**2) <= config.robot_radius:
        return True
    return False

def atGoal(config, x):
    if math.sqrt((x[0] - config.goalX)**2 + (x[1] - config.goalY)**2) <= config.robot_radius:
        return True
    return False

# Calculate goal cost via Pythagorean distance to robot
def calc_to_goal_cost(traj, config):
    
    # If-Statements to determine negative vs positive goal/trajectory position
    # traj[-1,0] is the last predicted X coord position on the trajectory
    if (config.goalX >= 0 and traj[-1,0] < 0):
        dx = config.goalX - traj[-1,0]
    elif (config.goalX < 0 and traj[-1,0] >= 0):
        dx = traj[-1,0] - config.goalX
    else:
        dx = abs(config.goalX - traj[-1,0])
    
    # traj[-1,1] is the last predicted Y coord position on the trajectory
    if (config.goalY >= 0 and traj[-1,1] < 0):
        dy = config.goalY - traj[-1,1]
    elif (config.goalY < 0 and traj[-1,1] >= 0):
        dy = traj[-1,1] - config.goalY
    else:
        dy = abs(config.goalY - traj[-1,1])

    # print("dx, dy", dx, dy)
    cost = math.sqrt(dx**2 + dy**2)
    # print("Cost: ", cost)
    return cost


def transform_traj_to_costmap(config, traj):
    # Get trajectory points wrt robot
    traj_odom = traj[:,0:2]
    traj_rob_x = (traj_odom[:, 0] - config.x)*math.cos(config.th) + (traj_odom[:, 1] - config.y)*math.sin(config.th)
    traj_rob_y = -(traj_odom[:, 0] - config.x)*math.sin(config.th) + (traj_odom[:, 1] - config.y)*math.cos(config.th)
    traj_norm_x = (traj_rob_x/config.costmap_resolution).astype(int)
    traj_norm_y = (traj_rob_y/config.costmap_resolution).astype(int)

    # Get traj wrt costmap
    traj_cm_col = config.costmap_shape[0]/2 - traj_norm_y
    traj_cm_row = config.costmap_shape[0]/2 - traj_norm_x

    return traj_cm_col, traj_cm_row

def calc_ref_path_follow_cost_1(config, traj):
    # Convert current waypoint wrt robot frame
    x_wp_rob , y_wp_rob = odom_to_robot(config, np.array([config.waypoint_curr[0]]), np.array([config.waypoint_curr[1]]))
    x_traj_rob , y_traj_rob = odom_to_robot(config, np.array([traj[-1, 0]]), np.array([traj[-1, 1]]))
    return abs(y_wp_rob - y_traj_rob) # * abs(x_wp_rob - x_traj_rob)


def calc_ref_path_follow_cost_2(config, traj):
    # If using ref path following with averaging
    x_ref_rob, y_ref_rob = odom_to_robot(config, np.array(config.x_ref_odom), np.array(config.y_ref_odom))
    x_traj_rob , y_traj_rob = odom_to_robot(config, np.array([traj[-1, 0]]), np.array([traj[-1, 1]]))

    config.x_ref_max = np.max(x_ref_rob)
    config.y_ref_avg = np.mean(y_ref_rob)
    
    
    return abs(config.y_ref_avg - y_traj_rob) # * abs(x_wp_rob - x_traj_rob)



def calc_obs_cost(config, traj, traj_cm_col, traj_cm_row):
    # NOTE: Planning costmap is set as glass_costmap_inflated
    config.planning_costmap = config.glass_costmap_inflated

    # Calculate cost based on distance to obstacle
    ob = np.argwhere(config.planning_costmap > config.obs_consideration_thresh) # (row, col) format
    # print("Size of obstacle set:", ob.shape)
    skip_n = 3
    minr = float("inf")

    # Loop through every obstacle in set and calc Pythagorean distance
    # Use robot radius to determine if collision
    for ii in range(0, len(traj[:, 1]), skip_n):
        for i in ob.copy():
            o_row = i[0]
            o_col = i[1]
            dx = traj_cm_col[ii] - o_col
            dy = traj_cm_row[ii] - o_row

            r = math.sqrt(dx**2 + dy**2)

            if r <= config.robot_radius:
                return float("Inf")  # collision

            if minr >= r:
                minr = r
    
    # -----------------------------------------------------------------------------------------------------------

    return 1.0 / minr



def draw_traj(config, traj, color):
    # print("Length of trajectory", len(traj))
    traj_array = np.asarray(traj)
    x_odom_list = np.asarray(traj_array[:, 0])
    y_odom_list = np.asarray(traj_array[:, 1])

    x_rob_list, y_rob_list = odom_to_robot(config, x_odom_list, y_odom_list)
    cm_col_list, cm_row_list = robot_to_costmap(config, x_rob_list, y_rob_list)

    costmap_traj_pts = np.array((cm_col_list.astype(int), cm_row_list.astype(int))).T

    costmap_traj_pts = costmap_traj_pts.reshape((-1, 1, 2))
    config.viz_costmap_inflated = cv2.polylines(config.viz_costmap_inflated, [costmap_traj_pts], False, color, 1)
    
    return config.viz_costmap_inflated




# NOTE: x_odom and y_odom are numpy arrays
def odom_to_robot(config, x_odom, y_odom):
    
    x_rob_odom_list = np.asarray([round(config.x, 2) for i in range(x_odom.shape[0])])
    y_rob_odom_list = np.asarray([round(config.y, 2) for i in range(y_odom.shape[0])])

    x_rob = (x_odom - x_rob_odom_list)*math.cos(config.th) + (y_odom - y_rob_odom_list)*math.sin(config.th)
    y_rob = -(x_odom - x_rob_odom_list)*math.sin(config.th) + (y_odom - y_rob_odom_list)*math.cos(config.th)

    return x_rob, y_rob


def robot_to_costmap(config, x_rob, y_rob):

    costmap_shape_list_0 = [config.costmap_shape[0]/2 for i in range(y_rob.shape[0])]
    costmap_shape_list_1 = [config.costmap_shape[1]/2 for i in range(x_rob.shape[0])]

    y_list = [int(y/config.costmap_resolution) for y in y_rob]
    x_list = [int(x/config.costmap_resolution) for x in x_rob]

    cm_col = np.asarray(costmap_shape_list_0) - np.asarray(y_list)
    cm_row = np.asarray(costmap_shape_list_1) - np.asarray(x_list)

    return cm_col, cm_row

class DwaNode(Node):
    def __init__(self):
        super().__init__('dwa_costmap')
        print(__file__ + " start!!")
        
        self.config = Config()
        self.obs = Obstacles()
        self.robot_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, 
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
            )
        # Subscriptions
        self.sub_odom = self.create_subscription(
            Odometry,
            '/odom',
            self.assign_odom_coords_callback,
            self.robot_qos_profile)
        
        self.sub_laser = self.create_subscription(
            LaserScan,
            '/scan',
            self.assign_laser_callback,
            10)
        
        self.sub_goal = self.create_subscription(
            Twist,
            '/target/position',
            self.target_callback,
            10)
        
        self.sub_ref_path = self.create_subscription(
            PoseArray,
            '/reference_path',
            self.ref_path_callback,
            10)
        
        # Lidar Intensity Map subscriber
        self.sub_intensity_map_mid = self.create_subscription(
            OccupancyGrid,
            '/intensity_map_mid',
            self.intensity_map_mid_callback,
            10)
        
        self.sub_intensity_map_low = self.create_subscription(
            OccupancyGrid,
            '/intensity_map_low',
            self.intensity_map_low_callback,
            10)
        
        self.sub_intensity_map_high = self.create_subscription(
            OccupancyGrid,
            '/intensity_map_high',
            self.intensity_map_high_callback,
            10)
        
        # Publishers
        choice = input("Publish? 1 or 0")
        if(int(choice) == 1):
            self.pub_cmd_vel = self.create_publisher(Twist, '/commands/velocity', 1)
            print("Publishing to /cmd_vel")
        else:
            self.pub_cmd_vel = self.create_publisher(Twist, '/dont_publish', 10)
            print("Not publishing!")

        # Visualization publisher
        self.config.viz_pub = self.create_publisher(
            sensor_msgs.msg.Image,
            '/viz_costmap',
            10)
        
        self.config.br = CvBridge()

        # Timer
        self.timer = self.create_timer(0.05, self.timer_callback)  # 20Hz
        
        # Initializations
        self.speed = Twist()
        self.x = np.array([self.config.x, self.config.y, self.config.th, 0.0, 0.0])
        self.u = np.array([0.0, 0.0])
        self.config.flag = 0

    def assign_odom_coords_callback(self, msg):
        self.config.assignOdomCoords(msg)

    def assign_laser_callback(self, msg):
        self.obs.assignObs(msg, self.config)

    def target_callback(self, msg):
        self.config.target_callback(msg)

    def ref_path_callback(self, msg):
        self.config.ref_path_callback(msg)

    def intensity_map_mid_callback(self, msg):
        self.config.intensity_map_mid_cb(msg)

    def intensity_map_low_callback(self, msg):
        self.config.intensity_map_low_cb(msg)

    def intensity_map_high_callback(self, msg):
        self.config.intensity_map_high_cb(msg)

    def timer_callback(self):
        # Main loop logic here, to be called every timer tick
        # Initial
    
        if self.config.goalX == 0.0006 and self.config.goalY == 0.0006:
            self.speed.linear.x = 0.0
            self.speed.angular.z = 0.0
            self.x = np.array([self.config.x, self.config.y, self.config.th, 0.0, 0.0])
        
        # Pursuing but not reached the goal
        elif (atGoal(self.config, self.x) == False): 
            if (len(self.config.ref_path) > 0) and (self.config.waypoint_curr is not None):
                if (crossed_current_waypoint(self.config, self.x) == True):
                    index = self.config.ref_path.index(self.config.waypoint_curr)
                    if index != 0:
                        self.config.waypoint_curr = self.config.ref_path[index-1]
                    else:
                        self.config.waypoint_curr = None
                        print("Reference path is over!")

            self.u = dwa_control(self.x, self.u, self.config, self.obs.obst)

            self.x[0] = self.config.x
            self.x[1] = self.config.y
            self.x[2] = self.config.th
            self.x[3] = self.u[0]
            self.x[4] = self.u[1]
            self.speed.linear.x = self.x[3]
            self.speed.angular.z = self.x[4]

        # If at goal then stay there until new goal published
        else:
            print("Goal reached!")
            self.speed.linear.x = 0.0
            self.speed.angular.z = 0.0
            self.x = np.array([self.config.x, self.config.y, self.config.th, 0.0, 0.0])
        
        self.config.viz_pub.publish(self.config.br.cv2_to_imgmsg(self.config.viz_costmap_inflated, encoding="bgr8"))
        self.pub_cmd_vel.publish(self.speed)


def dwa_control(x, u, config, ob):
    dw = calc_dynamic_window(x, config)
    u = calc_final_input(x, u, dw, config, ob)
    return u

def main(args=None):
    rclpy.init(args=args)
    dwa_node = DwaNode()
    rclpy.spin(dwa_node)
    dwa_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
