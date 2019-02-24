import os.path


class RosbagLoader():
    def __init__(self, base_dir):
        '''
        set base initializations
        '''
        self.base_dir = base_dir
        self.rosbag_dirs = [os.path.join(self.base_dir, d) for d in os.listdir(self.base_dir)
                            if os.path.isdir(os.path.join(self.base_dir, d))]
        self.mp_dirs = []

    def get_mp_dirs(self):
        for rosbag_dir in self.rosbag_dirs:
            mp_dirs_in_rosbag = [os.path.join(rosbag_dir, d) for d in os.listdir(rosbag_dir)
                                 if os.path.isdir(os.path.join(rosbag_dir, d))]

            print('rosbag_dir :', rosbag_dir)
            print(str(len(mp_dirs_in_rosbag)))
            if len(mp_dirs_in_rosbag) != 1:
                print("Two MP folder found in rosbag: {}".format(rosbag_dir))
            else:
                self.mp_dirs.extend(mp_dirs_in_rosbag)

        return self.mp_dirs
