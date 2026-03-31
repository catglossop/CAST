# Create a new tmux session
session_name="record_bag_$(date +%s)"
tmux new-session -d -s $session_name

# Split the window into three panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -h -p 50 # split it into two halves

# Run the roslaunch command in the first pane
tmux select-pane -t 0
tmux send-keys "source ~/create_ws/install/setup.bash" Enter
tmux send-keys "conda activate hi_learn" Enter
tmux send-keys "./launch_robot.sh" Enter  

# Change the directory to ../topomaps/bags and run the rosbag record command in the third pane
tmux select-pane -t 1
tmux send-keys "cd gold_standard_bags" Enter
tmux send-keys "ros2 bag record /front/image_raw /odom /cmd_vel -o $1" # change topic if necessary

# Attach to the tmux session
tmux -2 attach-session -t $session_name