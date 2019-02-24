from generator import GridMapGenerator
from rosbag_loader import RosbagLoader
import argparse
from helpers import timer, add_trailing_slash


@timer
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', metavar='I', type=str,
        help='The input folder path')
    args = parser.parse_args()
    base_dir = add_trailing_slash(args.i)

    print('base_dir :', base_dir)
    rl = RosbagLoader(base_dir)
    mp_dirs = rl.get_mp_dirs()

    gmg = GridMapGenerator(base_dir)
    gmg.start()

    '''for mp_dir in mp_dirs:
        print('mp_dir :', mp_dir)
        #gmg = GridMapGenerator(mp_dir)
        gmg = GridMapGenerator(base_dir)
        gmg.start()'''


if __name__ == "__main__":
    main()
