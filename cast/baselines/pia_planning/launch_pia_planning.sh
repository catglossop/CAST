#!/bin/bash

# Create a new tmux session
session_name="server_$(date +%s)"
tmux new-session -d -s $session_name

# Split the window into four panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -h -p 50  # split it into two halves
tmux selectp -t 1
tmux splitw -v -p 50

# Launch the robot
tmux select-pane -t 0
tmux send-keys "source ~/create_ws/install/setup.bash" Enter
tmux send-keys "conda activate hi_learn" Enter
tmux send-keys "../../deployment/launch/launch_robot.sh" Enter  

# Launch pia planning policy in the second pane
tmux select-pane -t 1
tmux send-keys "conda activate hi_learn" Enter
tmux send-keys "python pia_planning_policy.py $1" Enter

# Run the pd_controller.py script in the fourth pane
tmux select-pane -t 2
tmux send-keys "conda activate hi_learn" Enter
tmux send-keys "python ../../deployment/pd_controller.py" Enter


# Attach to the tmux session
tmux -2 attach-session -t $session_name
