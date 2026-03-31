#!/bin/bash

# Create a new tmux session
session_name="robot_launch_$(date +%s)"
# sudo chmod 666 /dev/ttyUSB0
tmux new-session -d -s $session_name

# Split the window into four panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -h -p 50 # split it into two halves
tmux selectp -t 0 
tmux splitw -v -p 50
tmux selectp -t 2
tmux splitw -v -p 50

# Launch the camera
tmux select-pane -t 0
tmux send-keys "conda activate hi_learn" Enter
tmux send-keys "source ~/create_ws/install/setup.bash" Enter
tmux send-keys "ros2 run usb_cam usb_cam_node_exe /image_raw:=/front/image_raw params_file:=../../config/camera.yaml" Enter

# Launch the joy controller
tmux select-pane -t 1
tmux send-keys "conda activate hi_learn" Enter
tmux send-keys "source ~/create_ws/install/setup.bash" Enter
tmux send-keys "ros2 launch teleop_twist_joy teleop-launch.py joy_vel:='commands/velocity' joy_config:=xbox" Enter

# Launch lidar
tmux select-pane -t 2
tmux send-keys "conda activate hi_learn" Enter
tmux send-keys "source ~/create_ws/install/setup.bash" Enter
tmux send-keys "ros2 launch rplidar_ros rplidar_s1_launch.py frame_id:='laser_link'" Enter

# Publish static transform
tmux select-pane -t 3
tmux send-keys "conda activate hi_learn" Enter
tmux send-keys "source ~/create_ws/install/setup.bash" Enter
tmux send-keys "ros2 launch kobuki_node kobuki_node-launch.py" Enter

# Attach to the tmux session
tmux -2 attach-session -t $session_name
