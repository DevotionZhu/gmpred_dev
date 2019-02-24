#! /bin/bash
source /home/mra/working/projects/ros2roh/bridge_demo/workspace/bridge_demo/devel/setup.bash

roscore & # Start the roscore initially

sleep 5s

# Folder where rosbags are stored
FILES=/home/mra/dataset/rosbags/2017_12_01/mp10/*

for f in $FILES
do

    # Create empty directory for each rosbag
    ROSBAG_PATH=$f
    ROS2ROH_PATH=/home/mra/dataset/rosbags/2017_12_01/mp10/$(basename $f .bag)/
    mkdir $ROS2ROH_PATH

    echo 'Extracting' $ROSBAG_PATH

    rosrun ros2roh ros2roh_node -d $ROS2ROH_PATH /mp10/SensorOutput,/mp10/pylon_camera_node_far/image_raw/,/mp10/pylon_camera_node_near/image_raw/ &
    sleep 10s
    rosbag play $ROSBAG_PATH

    sleep 21000s

    echo 'Finishing Ros2Roh processes'

    kill -SIGINT %2

    sleep 10s

done

echo 'Closing roscore'
kill -SIGINT %1
