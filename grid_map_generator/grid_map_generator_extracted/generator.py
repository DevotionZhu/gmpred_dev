from os import listdir, makedirs
from os.path import isfile, join, dirname, exists, basename
import shutil
import itertools
import xml.etree.ElementTree as ET
import numpy as np
import math
import scipy.misc
import cv2
from combiner import GridMapCombiner
from helpers import timer
from PIL import Image, ImageDraw
import colorsys #hsv to rgb transformation for radar colors
from collections import OrderedDict


class GridMapGenerator():
    def __init__(self, mp_dir):
        '''
        set parameters for the program
        '''
        # Parameters from caller
        self.out_base_path = join(mp_dir, 'processed/')
        self.mp_dir = mp_dir

        self.namespace = basename(mp_dir)

        # Set the namespace parameters
        self.set_namespace_parameters(self.namespace)

        # Output paths
        self.out_near_images = join(self.out_base_path, 'near_images/')
        self.out_far_images = join(self.out_base_path, 'far_images/')
        self.out_grid_images = join(self.out_base_path, 'grid_image/')
        self.out_np_images = join(self.out_base_path, 'numpy/')
        self.out_near_projected_radar_images = join(self.out_base_path, 'near_projected_radar_images/')
        self.out_far_projected_radar_images = join(self.out_base_path, 'far_projected_radar_images/')
        self.out_combined_images = join(self.out_base_path, 'combined/')
        self.grid_map_y_stretching_factor = 4

        self.handle_out_folder(self.out_base_path)
        self.handle_out_folder(self.out_near_images)
        self.handle_out_folder(self.out_far_images)
        self.handle_out_folder(self.out_grid_images)
        self.handle_out_folder(self.out_np_images)
        self.handle_out_folder(self.out_near_projected_radar_images)
        self.handle_out_folder(self.out_far_projected_radar_images)
        self.handle_out_folder(self.out_combined_images)

        # Input paths
        self.near_camera_dir_name = "pylon_camera_node_near/image_raw"
        self.far_camera_dir_name = "pylon_camera_node_far/image_raw"
        self.radar_data_dir_name = "SensorOutput/data.xml"

        # Grid Map Specs
        self.x_max_gridmap = 400
        self.x_min_gridmap = -50
        self.y_max_gridmap = 50
        self.y_max_gridmap_stretched = 50*self.grid_map_y_stretching_factor
        self.y_min_gridmap = -50
        self.grid_map_count = 1

        # combiner object
        self.gmc = GridMapCombiner(self.out_combined_images)

        # get points of lanelets
        self.lane_points = self.get_points()

        # image parameters
        self.combine_image_resize_ratio = 0.3
        self.image_size = (1200, 1920)

        self.write_config()

    def set_namespace_parameters(self, namespace):
        '''
        Set all the parameters that are related to namespace/measurement points
        '''
        if namespace == "mp09":
            self.roll_near = math.radians(-8.7)
            self.pitch_near = math.radians(1.7)
            self.yaw_near = math.radians(1)

            self.roll_far = math.radians(-0.7)
            self.pitch_far = math.radians(-0.2)
            self.yaw_far = math.radians(-0.2)

            # Should be the same as height_pylon_far
            self.height_radar = 7.13+0.375


            self.K_near = np.array([[4211.055664, 0.0, 917.65746, 0.0], [0.0, 4252.427734, 564.533056, 0.0], [0.0, 0.0, 1.0, 0.0]])
            self.K_far = np.array([[13067.078125, 0.0, 946.848798, 0.0], [0.0, 13050.034180, -16.105275, 0.0], [0.0, 0.0, 1.0, 0.0]])
        else:
            self.roll_near = math.radians(-7.8)
            self.pitch_near = math.radians(-0.6)
            self.yaw_near = math.radians(-1)

            self.roll_far = math.radians(-0.8)
            self.pitch_far = math.radians(3.1)
            self.yaw_far = math.radians(-1)

            # Should be the same as height_pylon_far
            self.height_radar = 6.53+0.375

            self.K_near = np.array([[4250.36084, 0.0, 884.465788, 0.0], [0.0, 4285.873047, 528.032106, 0.0], [0.0, 0.0, 1.0, 0.0]])
            self.K_far = np.array([[13140.333984, 0.0, 968.70729, 0.0], [0.0, 13127.654297, 140.872204, 0.0], [0.0, 0.0, 1.0, 0.0]])

        self.translation_vector_near = np.array([-0.193, 0.123, self.height_radar])
        self.translation_vector_far = np.array([-0.193, -0.07, self.height_radar])
        self.h_gt_near = self.create_hgt(self.roll_near, self.pitch_near, self.yaw_near, self.translation_vector_near)
        self.h_gt_far = self.create_hgt(self.roll_far, self.pitch_far, self.yaw_far, self.translation_vector_far)

    def handle_out_folder(self, directory):
        '''
        See if folder exists: If yes, remove it. Then create directory.
        '''
        if exists(directory):
            shutil.rmtree(directory)
        makedirs(directory)

    def get_image_paths(self, directory):
        '''
        Get all image paths from a given directory
        '''
        file_paths = [join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]
        return file_paths

    def read_radar_data(self, directory):
        '''
        Reads raw XML radar data
        '''
        f = open(directory)
        try:
            it = itertools.chain('<root>', f, '</root>')
            root = ET.fromstringlist(it)
        except:
            f.close()
            f = open(directory)
            # Maybe its an incomplete XML
            it = itertools.chain('<root>', f, '<dimensions><_1>4.6</_1> <_2>0</_2> <_3>0</_3></dimensions>'
                                              '</object>'
                                              '</object_list>'
                                              '</SensorOutput>'
                                              '</data>'
                                              '</root>')
            root = ET.fromstringlist(it)

        f.close()
        return root.find('data')

    def write_config(self):
        '''
        Writes text file that contains configuration info for the entire processing run
        '''
        f = open(self.out_base_path + "configuration.txt", "w")
        f.write("**************GENERAL************************" + '\n')
        f.write("namespace: " + str(self.namespace) + '\n')
        f.write("mp_dir: " + str(self.mp_dir) + '\n')
        f.write("out_base_path: " + str(self.out_base_path) + '\n')
        f.write("**************NEAR CAMERA PARAMETERS************************" + '\n')
        f.write("roll_near: " + str(self.roll_near) + '\n')
        f.write("pitch_near: " + str(self.pitch_near) + '\n')
        f.write("yaw_near: " + str(self.yaw_near) + '\n')
        f.write("translation_vector near: " + str(self.translation_vector_near) + '\n')
        f.write("K_near: " + str(self.K_near) + '\n')
        f.write("h_gt_near: " + str(self.h_gt_near) + '\n')
        f.write("**************FAR CAMERA PARAMETERS************************" + '\n')
        f.write("roll_far: " + str(self.roll_far) + '\n')
        f.write("pitch_far: " + str(self.pitch_far) + '\n')
        f.write("yaw_far: " + str(self.yaw_far) + '\n')
        f.write("height_radar: " + str(self.height_radar) + '\n')
        f.write("translation_vector far: " + str(self.translation_vector_far) + '\n')
        f.write("K_far: " + str(self.K_far) + '\n')
        f.write("h_gt_far: " + str(self.h_gt_far) + '\n')
        f.write("**************GRID MAP RESOLUTION************************" + '\n')
        f.write("x_max_gridmap: " + str(self.x_max_gridmap) + '\n')
        f.write("x_min_gridmap: " + str(self.x_min_gridmap) + '\n')
        f.write("y_max_gridmap: " + str(self.y_max_gridmap) + '\n')
        f.write("y_min_gridmap: " + str(self.y_min_gridmap) + '\n')
        f.write("Grid Map Rows: " + str(self.x_max_gridmap + abs(self.x_min_gridmap)) + '\n')
        f.write("Grid Map Cols: " + str(self.y_max_gridmap + abs(self.y_min_gridmap)) + '\n')
        f.write("**************GRID MAP IMAGE RESOLUTION************************" + '\n')
        f.write("Grid Map Image Height: " + str(self.x_max_gridmap + abs(self.x_min_gridmap)) + '\n')
        f.write("Grid Map Image Width: " + str(self.y_max_gridmap_stretched + abs(self.y_min_gridmap)) + '\n')
        f.write("**************NEAR/FAR IMAGE RESOLUTION************************" + '\n')
        f.write("Image Height: " + str(self.image_size[0]) + '\n')
        f.write("Image Width: " + str(self.image_size[1]) + '\n')
        f.write("**************COMBINED IMAGE RESOLUTION************************" + '\n')
        f.write("Grid Map Image Height: " + str(self.x_max_gridmap + abs(self.x_min_gridmap)) + '\n')
        f.write("Grid Map Image Width: " + str(self.y_max_gridmap_stretched + abs(self.y_min_gridmap)) + '\n')
        f.write("Image Height: " + str(int(self.combine_image_resize_ratio*self.image_size[0])) + '\n')
        f.write("Image Width: " + str(int(self.combine_image_resize_ratio*self.image_size[1])) + '\n')
        f.write("Combined Image Height: " + str(self.x_max_gridmap +
                                                abs(self.x_min_gridmap)) + '\n')
        f.write("Combined Image Width: " + str(int(self.combine_image_resize_ratio*self.image_size[1]
                                               +self.y_max_gridmap_stretched +
                                               abs(self.y_min_gridmap))) + '\n')
        f.close()

    def do_point_transformation(self, x, y):
        y = y * -1
        x = abs(int(x) - self.x_max_gridmap)
        y = int(y) - self.y_min_gridmap
        return x, y

    def do_point_transformation_stretched(self, x, y):
        y = y * self.grid_map_y_stretching_factor
        y -= 15 * self.grid_map_y_stretching_factor  # slight shift to left
        return x, y

    def get_points_from_lane(self, lane):
        points = []
        for point in lane.findall('point'):
            x = float(point.find('x').text)
            y = float(point.find('y').text)

            x, y = self.do_point_transformation(x, y)
            x, y = self.do_point_transformation_stretched(x, y)
            points.append((x, y))

        return points

    def get_points(self):
        lane_points = []
        e = ET.parse('lanelets.txt').getroot()
        for lanelet in e.findall('lanelet'):
            leftbound = lanelet.find('leftBound')
            rightbound = lanelet.find('rightBound')
            lane_points.append(self.get_points_from_lane(leftbound))
            lane_points.append(self.get_points_from_lane(rightbound))
        return lane_points

    def draw_line(self, image, x1, y1, x2, y2):
        draw = ImageDraw.Draw(image)
        # draw.line((x2, y2, x1, y1), fill=150)
        draw.line((y2, x2, y1, x1), fill=150)

        image.save('temp.jpeg')

        return image

    def draw_lanes(self, image):
        for lane in self.lane_points:
            for i in range(0, len(lane) - 1):
                point = lane[i]
                x1 = point[0]
                y1 = point[1]

                point_1 = lane[i + 1]

                x2 = point_1[0]
                y2 = point_1[1]

                image = self.draw_line(image, x1, y1, x2, y2)
                # image.show()

        # image.show()
        return image


    def create_rotation_matrix(self, roll_angle, pitch_angle, yaw_angle):
        '''
        Create a rotation matrix for given roll, pitch and yaw angles.
        Create rotation matrices for x, y and z coordinates and multiply them
        '''
        #x
        roll_c, roll_s = np.cos(roll_angle), np.sin(roll_angle)
        roll_matrix = np.array([[1, 0, 0], [0, roll_c, roll_s], [0, -roll_s, roll_c]])
        #y
        pitch_c, pitch_s = np.cos(pitch_angle), np.sin(pitch_angle)
        pitch_matrix = np.array([[pitch_c, 0, -pitch_s], [0, 1, 0], [pitch_s, 0, pitch_c]])
        #z
        yaw_c, yaw_s = np.cos(yaw_angle), np.sin(yaw_angle)
        yaw_matrix = np.array([[yaw_c, yaw_s, 0], [-yaw_s, yaw_c, 0], [0, 0, 1]])

        return (roll_matrix.dot(pitch_matrix)).dot(yaw_matrix)


    def create_hgt(self, roll, pitch, yaw, translation_vector):
        '''
        Create a H_groundtruth matrix (Transformation matrix that contains rotation and translation)
        '''
        rotation_basic = np.array([[0, -1, 0],[0, 0, -1],[1, 0, 0]]) #from radar to camera coordiante system
        rotation_matrix=(self.create_rotation_matrix(roll, pitch, yaw)).dot(rotation_basic)

        h_gt = np.identity(4)
        h_gt[:3, :3] = rotation_matrix
        translation_vector=-rotation_matrix.dot(translation_vector)
        h_gt[:3, 3] = translation_vector

        return h_gt


    def get_uv_invdepth(self, K, h_gt, point):
        '''
        get pixels and depth
        '''
        tmp = K.dot(h_gt)
        #print('get_uv_invdepth-> tmp = K.dot(h_gt):' + str(tmp))
        point=tmp.dot(point)
        #print('get_uv_invdepth-> point=tmp.dot(point):' + str(point))

        if point[2] != 0:
            return np.array([int(point[0]/point[2]), int(point[1]/point[2]), 1/point[2]])
        else:
            return None


    def project_radar_on_image(self, img, radar_x, radar_y, K, h_gt):
        '''
        Projects radar coordinates onto image abd draws circles,
        '''
        #print('project_radar_on_image> k:' + str(K) + ' h_gt:' + str(h_gt) + ' radar_x:' + str(radar_x) + ' radar_y:' + str(radar_y) )
        (u, v, inv_depth) = self.get_uv_invdepth(K, h_gt, [radar_x, radar_y, 0, 1])
        return u, v


    def add_marker_image(self, img, point, marker, fontscale, linetype):
        '''
        Adds some text to a point in an image,
        '''
        font                    = cv2.FONT_HERSHEY_SIMPLEX
        bottom_left_corner      = point
        font_scale              = fontscale
        font_color              = (0,0,0)
        line_type               = linetype

        cv2.putText(img, marker,
                    bottom_left_corner,
                    font,
                    font_scale,
                    font_color,
                    line_type)


    def convert_to_rgb(self, val):
        '''
        Convert val into a color in rbg
        '''
        color = tuple(round(i * 255) for i in colorsys.hsv_to_rgb(val*10/36, 1, 1))
        return color



    def add_circle_image(self, img, point, circle_radius, object_marker_id):
        #cv2.circle(img, point, circle_radius, (0,0,255), -1) # without tracker colors related to object id
        marker_color = self.convert_to_rgb(float(object_marker_id))
        cv2.circle(img, point, circle_radius, marker_color, -1)

    def write_image_disk(self, path, img):
        cv2.imwrite(path, img)

    def process_data(self, img_path, radar_data, out_images, out_projected_radar_images, grid_map_count, K, h_gt, process_grid=True):
        '''
        Callback function that gets every message of camera and radar,
        '''
        # if(abs(float(ros_img.header.stamp.nsecs-radar_data.header.stamp.nsecs)/1000000000)<0.5):

        #print("process_data> out_images: ", out_images)
        #print("process_data> out_projected_radar_images: ", out_projected_radar_images)
        #print('process_data> img_path:', img_path)
        #print("process_data> process_grid: ", process_grid)
        if img_path:
            # Prepare near and far images
            radar_wo_camera_image = np.ones((self.image_size[0], self.image_size[1], 3), np.uint8)*255
            radar_with_camera_image = cv2.imread(img_path, 1)
            # Make empty grid map
            grid_map = np.zeros((self.x_max_gridmap + abs(self.x_min_gridmap),
                                self.y_max_gridmap + abs(self.y_min_gridmap)),
                               dtype=int)

            # Grid map image representation
            grid_map_image = np.ones((self.x_max_gridmap + abs(self.x_min_gridmap),
                                      self.y_max_gridmap_stretched + abs(self.y_min_gridmap), 3),
                                     np.uint8) * 255

            # Iterate through all the tracked objects
            for radar_object in radar_data['object_list']:
                # Resolve object ID to our own ID
                # object_id = self.resolve_id_to_text(d.object_ID)
                object_id = radar_object.find("id").text
                object_position = radar_object.find("position")
                pos_1 = float(object_position.find("_1").text)
                pos_2 = float(object_position.find("_2").text)

                # Handle camera image
                u, v = self.project_radar_on_image(radar_with_camera_image, pos_1, pos_2, K, h_gt)
                #self.add_marker_image(radar_with_camera_image, (int(u),int(v)), object_id, 2, 3)
                self.add_circle_image(radar_with_camera_image, (int(u),int(v)), 10, object_id)
                #self.add_marker_image(radar_wo_camera_image, (int(u),int(v)), object_id, 1, 2) #Outcomment to not place text markers on pure 3d radar projections
                self.add_circle_image(radar_wo_camera_image, (int(u),int(v)), 10, object_id)

                if pos_1 <= self.x_max_gridmap and pos_2 >= self.y_min_gridmap:
                    x, y = self.do_point_transformation(pos_1, pos_2)
                    x_stretched, y_stretched = self.do_point_transformation_stretched(x, y)
                    try:
                        grid_map[x][y] = 1
                        self.add_marker_image(grid_map_image, (y_stretched, x_stretched), object_id, 0.3, 1)
                        self.add_circle_image(grid_map_image, (y_stretched, x_stretched), 2, object_id)
                    except Exception as e:
                        print("Grid Map error: {}".format(str(e)))


                    self.add_marker_image(grid_map_image, (y, x), object_id, 0.3, 1)
                    self.add_circle_image(grid_map_image, (y, x), 2, object_id)


            radar_ts_s = radar_data['s']
            radar_ts_ns = radar_data['ns']
            image_ts = basename(img_path).replace(".png", "")
            image_ts_s = image_ts.split("_")[0]
            image_ts_ns = image_ts.split("_")[1]

            # Near: Saving projected camera + radar images
            self.write_image_disk(out_images +
                                  "{}-{}_{}-{}_{}.png".format(grid_map_count, image_ts_s, image_ts_ns,
                                                              radar_ts_s, radar_ts_ns), radar_with_camera_image)
            # Near: Saving projected radar only images
            self.write_image_disk(out_projected_radar_images +
                                  "{}-{}_{}-{}_{}.png".format(grid_map_count, image_ts_s, image_ts_ns,
                                                              radar_ts_s, radar_ts_ns), radar_wo_camera_image)
            if process_grid:
                # Saving the 'numpy' grid_map
                np.save(self.out_np_images + "{}-{}_{}-{}_{}".format(grid_map_count, image_ts_s, image_ts_ns,
                                                                         radar_ts_s, radar_ts_ns), grid_map)
                #print("After saving np array -> out_images:", out_images)
                print('saved numpy array no.: ' + str(grid_map_count))
                # Saving the grid map image representation
                # grid_map_image = cv2.resize(grid_map_image, None, fx=1.8, fy=1, interpolation=cv2.INTER_CUBIC)
                grid_map_image = Image.fromarray(grid_map_image.astype('uint8'), 'RGB')
                # grid_map_image = self.draw_lanes(grid_map_image)
                grid_map_image.save(self.out_grid_images + "{}-{}_{}-{}_{}.png".format(grid_map_count, image_ts_s,
                                                                                       image_ts_ns, radar_ts_s, radar_ts_ns))
                # Make combined image

                # make grid map image bigger
                # resize_ratio = 2
                # i1_re = grid_map_image.resize([int(resize_ratio * s) for s in grid_map_image.size],
                #                   Image.ANTIALIAS)
                i1_re = grid_map_image

                # make camera image a bit smaller
                radar_with_camera_image = Image.fromarray(cv2.cvtColor(radar_with_camera_image, cv2.COLOR_BGR2RGB))
                i2_re = radar_with_camera_image.resize([int(self.combine_image_resize_ratio * s)
                                                        for s in radar_with_camera_image.size],
                                                        Image.ANTIALIAS)
                self.gmc.do_combine(i2_re, i1_re, "{}-{}_{}-{}_{}.png".format(grid_map_count, image_ts_s, image_ts_ns,
                                                                          radar_ts_s, radar_ts_ns))

            # print ("Time for radar_callback: {}".format(data.header.stamp))



    def preprocess_image_paths(self, image_paths):
        image_preprocess = {}
        for image_path in image_paths:
            image_ts = basename(image_path).replace(".png", "")
            image_ts_s = image_ts.split("_")[0]
            image_ts_ns = image_ts.split("_")[1]

            image_preprocess[image_path] = float(image_ts_s) + float(image_ts_ns)/1000000000
            #print('preprocess_image_paths:' + image_preprocess[image_path])

        #return image_preprocess
        ordered_dict = OrderedDict(sorted(image_preprocess.items(), key=lambda t: t[1]))
        #print('preprocess_image_paths:' + str(ordered_dict))
        return ordered_dict

    def find_closest_image(self, radar_object, image_paths_preprocessed):
        threshold = 0.079
        radar_ts_s = radar_object['s']
        radar_ts_ns = radar_object['ns']
        radar_ts = float(radar_ts_s) + float(radar_ts_ns)/1000000000
        nearest_image = None
        closest_ts = 100000000000
        for image_path, image_ts in image_paths_preprocessed.items():
            image_ts_adj = image_ts + 237/1000  # Added delay for camera
            if abs(radar_ts - image_ts_adj) < closest_ts:
                closest_ts = abs(radar_ts - image_ts_adj)
                nearest_image = image_path

        #if closest_ts < threshold:
        return nearest_image

        #return None

    def sort_radar_key(self, x):
        return float(x['s'] + "." + x['ns'])

    def start(self):
        '''
        Main function that does all the processing,
        '''
        print("Processing {}".format(self.mp_dir))

        radar_data_root = self.read_radar_data(join(self.mp_dir, self.radar_data_dir_name))
        near_images_paths = self.get_image_paths(join(self.mp_dir, self.near_camera_dir_name))
        far_images_paths = self.get_image_paths(join(self.mp_dir, self.far_camera_dir_name))
        image_paths_preprocessed_near = self.preprocess_image_paths(near_images_paths)
        image_paths_preprocessed_far = self.preprocess_image_paths(far_images_paths)

        #print(image_paths_preprocessed_near.values())
        near_image_keys = list(image_paths_preprocessed_near.keys())
        print('near_image_keys: '+ str(len(near_image_keys)))
        near_image_values = list(image_paths_preprocessed_near.values())
        print('near_image_values: '+ str(len(near_image_values)))
        #print('{0:.7f}'.format(near_image_values[0]))
        #print('{0:.7f}'.format(near_image_values[1]))
        #print(near_image_keys[0])
        #print(near_image_keys[1])

        grid_map_count_near = 0
        grid_map_count_far = 0

        radar_data = []
        for radar_object in radar_data_root.findall('SensorOutput'):
            radar_data_obj = {}
            radar_ts_s = radar_object.find('timestamp').find('s').text
            radar_ts_ns = radar_object.find('timestamp').find('ns').text
            radar_data_obj['s'] = radar_ts_s
            radar_data_obj['ns'] = radar_ts_ns
            radar_data_obj['object_list'] = radar_object.find("object_list").findall("object")
            radar_data.append(radar_data_obj)

        radar_data.sort(key=lambda x: (int(x['s']), int(x['ns'])))
        print('radar_data array length: ' + str(len(radar_data)))

        index = 0
        for radar_object in radar_data:
            image_path = near_image_keys[index]
            index += 1
            #nearest_image_path = self.find_closest_image(radar_object, image_paths_preprocessed_near)
            #print('nearest image path:' + nearest_image_path)
            #self.process_data(nearest_image_path, radar_object, self.out_near_images, self.out_near_projected_radar_images, grid_map_count_near, self.K_near, self.h_gt_near)
            self.process_data(image_path, radar_object, self.out_near_images, self.out_near_projected_radar_images, grid_map_count_near, self.K_near, self.h_gt_near)

            #far_image_path = self.find_closest_image(radar_object, image_paths_preprocessed_far)
            #self.process_data(far_image_path, radar_object, self.out_far_images, self.out_far_projected_radar_images, grid_map_count_far, self.K_far, self.h_gt_far, process_grid=False)

            grid_map_count_near += 1
            grid_map_count_far += 1

        print("Completed")
