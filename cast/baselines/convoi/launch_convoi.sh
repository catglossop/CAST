#!/bin/bash

# Create a new tmux session
session_name="task_$(date +%s)"
tmux new-session -d -s $session_name


# Split the window into four panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -h -p 50 # split it into two halves
tmux selectp -t 0    # select the first (0) pane
tmux splitw -v -p 50 # split it into two halves
tmux selectp -t 2
tmux splitw -v -p 50
tmux selectp -t 3
tmux splitw -h -p 50

# Launch the robot
tmux select-pane -t 0
tmux send-keys "source ~/create_ws/install/setup.bash" Enter
tmux send-keys "conda activate hi_learn" Enter
tmux send-keys "../../deployment/launch/launch_robot.sh" Enter  

# Run the pd_controller.py script in the fourth pane
tmux select-pane -t 1
tmux send-keys "conda activate hi_learn" Enter
tmux send-keys "source ~/create_ws/install/setup.bash" Enter
tmux send-keys "python planner_ros2.py" Enter

# Launch the gpt node
tmux select-pane -t 2
tmux send-keys "conda activate hi_learn" Enter
tmux send-keys "source ~/create_ws/install/setup.bash" Enter
tmux send-keys "python gpt-v5-ros2-2.py $@" Enter

# Launch occupancy grid publishing
tmux select-pane -t 3
tmux send-keys "conda activate hi_learn" Enter
tmux send-keys "source ~/create_ws/install/setup.bash" Enter
tmux send-keys "ros2 run nav2_costmap_2d nav2_costmap_2d --ros-args --params-file /home/create/create_ws/src/vlm-guidance/config/nav2_params.yaml " Enter

# Start publishing a map to odom transform
tmux select-pane -t 4
tmux send-keys "conda activate hi_learn" Enter
tmux send-keys "source ~/create_ws/install/setup.bash" Enter
tmux send-keys "ros2 lifecycle set /costmap/costmap configure" Enter 
tmux send-keys "ros2 lifecycle set /costmap/costmap activate" Enter 
tmux send-keys "ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 1 0 base_link laser_link" Enter

tmux select-pane -t 1
# Attach to the tmux session
tmux -2 attach-session -t $session_name