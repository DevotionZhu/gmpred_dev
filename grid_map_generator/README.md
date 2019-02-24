# grid_map_generator_extracted

## How to run
```bash _
python grid_map_generator_extracted/main.py -i /path/to/main_folder
```


## Notes
The /path/to/main_folder should be structured with folders with the structure /rosbag_name/mpXX/* [Output of ros2roh]


The combined image name is 'id-camerasecond_camerananseconds-radarsecond_radarnanoseconds.png'
# grid_map_ros [Depracated: Use grid_map_generator_extracted]

## Implementation of Grid Map for scene classification and prediction

The main code resides in the `grid_map_generator` directory

## Installing dependencies

1. catkin_pkg	`sudo apt-get install catkin_pkg`
2. multisense	`sudo apt-get install ros-kinetic-multisense`
3. doxygen		`sudo apt-get install doxigen`

## How to run
```bash
source /opt/ros/kinetic/setup.bash
catkin_make
source devel/setup.bash
roslaunch grid_map_generator grid_map_generator.launch
```

In the launch file you need to specify some parameters via the command line:

| Arg        | Value           | 
| ------------- |-------------|
| `rosbag`      | Path to your rosbag |
|  out_folder   | output directory where everything is outputted, should be a directory      

According to your rosbag, the following parameters should also be set:

```xml
<arg name="mp09" default="true" />
<arg name="mp10" default="false" />
```

# grid_map_classifier
Not complete - deep learning code for classification of scenarios in grid map