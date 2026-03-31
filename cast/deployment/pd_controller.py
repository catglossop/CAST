import numpy as np
import yaml
from typing import Tuple

# ROS
from topic_names import (WAYPOINT_TOPIC, 
			 			REACHED_GOAL_TOPIC)
import rclpy
from rclpy.node import Node
from ros_data import ROSData

from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Bool

CONFIG_PATH = "/home/create/create_ws/src/vlm-guidance/config/robot.yaml"
CMD_VEL_TOPIC = "/commands/velocity"
class PDController(Node): 

    def __init__(self, 
                 config_path, 
                 ):
        super().__init__('pd_controller')
        self.load_config(config_path)

        self.vel_msg = Twist()
        self.waypoint = ROSData(self.WAYPOINT_TIMEOUT, name="waypoint")
        self.reached_goal = False
        self.reverse_mode = False
        self.current_yaw = None

        self.waypoint_sub = self.create_subscription(
            Float32MultiArray,
            WAYPOINT_TOPIC,
            self.waypoint_callback,
            1)
        self.reached_goal_sub = self.create_subscription(
            Bool, 
            REACHED_GOAL_TOPIC, 
            self.reached_goal_callback, 
            1)
        self.vel_out = self.create_publisher(
            Twist, 
            self.VEL_TOPIC, 
            1)
        self.timer_period = 0.01  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
    
    def load_config(self, config_path):
        with open(config_path, "r") as f:
            robot_config = yaml.safe_load(f)
        self.MAX_V = robot_config["max_v"]
        self.MAX_W = robot_config["max_w"]
        self.VEL_TOPIC = CMD_VEL_TOPIC 
        self.DT = 1/robot_config["frame_rate"]
        self.RATE = robot_config["frame_rate"]
        self.EPS = 1e-8
        self.WAYPOINT_TIMEOUT = 1 # seconds # TODO: tune this


    def clip_angle(self, theta) -> float:
        """Clip angle to [-pi, pi]"""
        theta %= 2 * np.pi
        if -np.pi < theta < np.pi:
            return theta
        return theta - 2 * np.pi
        

    def pd_controller(self, waypoint: np.ndarray) -> Tuple[float]:
        """PD controller for the robot"""
        assert len(waypoint) == 2 or len(waypoint) == 4, "waypoint must be a 2D or 4D vector"
        if len(waypoint) == 2:
            dx, dy = waypoint
        else:
            dx, dy, hx, hy = waypoint
        print("waypoint: ", dx, dy)
        # this controller only uses the predicted heading if dx and dy near zero
        if len(waypoint) == 4 and np.abs(dx) < self.EPS and np.abs(dy) < self.EPS:
            v = 0
            w = self.clip_angle(np.arctan2(hy, hx))/self.DT		
        elif np.abs(dx) < self.EPS:
            v =  0
            w = np.sign(dy) * np.pi/(2*self.DT)
        else:
            v = dx / self.DT
            w = np.arctan(dy/dx) / self.DT
        v = np.clip(v, 0, self.MAX_V)
        w = np.clip(w, -self.MAX_W, self.MAX_W)
        return v, w


    def waypoint_callback(self, waypoint_msg: Float32MultiArray):
        """Callback function for the waypoint subscriber""" 
        # TODO: add config for different kinds of waypoints/goals?
        print("Setting waypoint")
        self.waypoint.set(waypoint_msg.data)
        
        
    def reached_goal_callback(self, reached_goal_msg: Bool):
        """Callback function for the reached goal subscriber"""
        print("Checking goal reached")
        self.reached_goal = reached_goal_msg.data
    
    def timer_callback(self):

        if self.reached_goal:
            self.vel_out.publish(self.vel_msg)
            print("Reached goal! Stopping...")
            return
        elif self.waypoint.is_valid(verbose=True):
            v, w = self.pd_controller(self.waypoint.get())
            if self.reverse_mode: 
                v *= -1
            self.vel_msg.linear.x = v
            self.vel_msg.angular.z = w
            print(f"publishing new vel: {v}, {w}")
        self.vel_out.publish(self.vel_msg)       


def main(args=None):
    rclpy.init()
    print("Registered with master node. Waiting for waypoints...")

    pd_controller_node = PDController(CONFIG_PATH)

    rclpy.spin(pd_controller_node)
    pd_controller_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()