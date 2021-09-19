import os
import argparse
import pickle

import tqdm

import context
from visualdet3d.utils import Timer
from visualdet3d.data.kitti.preprocessing import KittiData
from configs import load_config


def read_one_split(cfg,
                   index_names,
                   data_root_dir,
                   output_dict,
                   data_split='training',
                   time_display_inter=100):
    """
    """
    N = len(index_names)
    frames = [None] * N
    print(f'Start reading {data_split} data')
    timer = Timer()

    for i, index_name in tqdm.tqdm(enumerate(index_names)):

        # read data with dataloader api
        data_frame = KittiData(data_root_dir, index_name, output_dict)
        calib, _, _, _ = data_frame.read_data()

        # store the list of kittiObjet and kittiCalib
        data_frame.calib = calib

        frames[i] = data_frame

        if (i+1) % time_display_inter == 0:
            avg_time = timer.compute_avg_time(i+1)
            eta = timer.compute_eta(i+1, N)
            print("{} iter:{}/{}, avg-time:{}, eta:{}".format(
                data_split, i+1, N, avg_time, eta), end='\r')

    save_dir = os.path.join(cfg.path.preprocessed_path, data_split)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    pkl_file = os.path.join(save_dir, 'imdb.pkl')
    pickle.dump(frames, open(pkl_file, 'wb'))
    print(f'{data_split} split finished precomputing')


def main():
    cfg = load_config(args.config)

    time_display_inter = 100
    data_root_dir = args.data_dir
    calib_path = os.path.join(data_root_dir, 'calib')
    list_calib = os.listdir(calib_path)
    N = len(list_calib)
    # no need for image, could be modified for extended use
    output_dict = {
        'calib': True,
        'image': False,
        'label': False,
        'velodyne': False,
    }

    num_test_file = N
    test_names = ["%06d" % i for i in range(num_test_file)]
    read_one_split(cfg, test_names, data_root_dir, output_dict,
                   'test', time_display_inter)

    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data-dir', type=str, help='Test data directory')
    parser.add_argument('--use_point_cloud', action='store_true',
                        help='Use point cloud')
    args = parser.parse_args()
    print(args)
    main()