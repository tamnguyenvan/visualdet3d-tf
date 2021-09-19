"""
"""
import os
import argparse
from typing import List, Dict
from copy import deepcopy

import cv2
import numpy as np
import skimage.measure
from tqdm import tqdm

import context
from visualdet3d.data.kitti.preprocessing import KittiData
from visualdet3d.data.pipeline.transforms import get_transform
from visualdet3d.data.kitti.utils import generate_dispariy_from_velo
from configs import load_config


def denorm(image, rgb_mean, rgb_std):
    """Denormalize a image.

    Args:
        image: np.ndarray normalized [H, W, 3]
        rgb_mean: np.ndarray [3] among [0, 1] image
        rgb_std : np.ndarray [3] among [0, 1] image
    Returns:
        unnormalized image: np.ndarray (H, W, 3) [0-255] dtype=np.uint8
    """
    image = image * rgb_std + rgb_mean #
    image[image > 1] = 1
    image[image < 0] = 0
    image *= 255
    return np.array(image, dtype=np.uint8)


def load_names(data_dir):
    """
    """
    train_list_file = os.path.join(data_dir, 'training.txt')
    train_image_indices = []
    with open(train_list_file) as f:
        for line in f:
            train_image_indices.append(line.strip())
    
    val_list_file = os.path.join(data_dir, 'validation.txt')
    val_image_indices = []
    with open(val_list_file) as f:
        for line in f:
            val_image_indices.append(line.strip())
    return train_image_indices, val_image_indices


def compute_dispairity_for_split(cfg,
                                 index_names: List[str], 
                                 data_root_dir: str, 
                                 output_dict: Dict, 
                                 data_split: str='training', 
                                 use_point_cloud: bool=True):
    """
    """
    save_dir = os.path.join(cfg.path.preprocessed_path, data_split)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    disp_dir = os.path.join(save_dir, 'disp')
    if not os.path.isdir(disp_dir):
        os.mkdir(disp_dir)

    if not use_point_cloud:
        stereo_matcher = cv2.StereoBM_create(192, 25)

    print(f'Reading {data_split} data...')
    transform = get_transform(cfg.data.augmentation.test)

    for i, index_name in tqdm(enumerate(index_names)):
        # Read data with dataloader api
        data_frame = KittiData(data_root_dir, index_name, output_dict)
        calib, image, right_image, label, velo = data_frame.read_data()

        original_image = image.copy()
        baseline = (calib.P2[0, 3] - calib.P3[0, 3]) / calib.P2[0, 0]
        image, image_3, P2, P3 = transform(original_image, right_image.copy(), p2=deepcopy(calib.P2), p3=deepcopy(calib.P3))
        if use_point_cloud:
            # Gathering disparity with point cloud back projection
            disparity_left = generate_dispariy_from_velo(
                velo[:, 0:3],
                image.shape[0],
                image.shape[1],
                calib.Tr_velo_to_cam,
                calib.R0_rect,
                P2,
                baseline=baseline
            )
            disparity_right = generate_dispariy_from_velo(
                velo[:, 0:3],
                image.shape[0],
                image.shape[1],
                calib.Tr_velo_to_cam,
                calib.R0_rect,
                P3,
                baseline=baseline
            )

        else:
            # Gathering disparity with stereoBM from opencv
            left_image  = denorm(image,
                                 cfg.data.augmentation.rgb_mean,
                                 cfg.data.augmentation.rgb_std)
            right_image = denorm(image_3,
                                 cfg.data.augmentation.rgb_mean,
                                 cfg.data.augmentation.rgb_std)
            gray_image1 = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
            gray_image2 = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

            disparity_left = stereo_matcher.compute(gray_image1, gray_image2)
            disparity_left[disparity_left < 0] = 0
            disparity_left = disparity_left.astype(np.uint16)

            disparity_right = stereo_matcher.compute(gray_image2[:, ::-1], gray_image1[:, ::-1])
            disparity_right[disparity_right < 0] = 0
            disparity_right= disparity_right.astype(np.uint16)

        disparity_left = skimage.measure.block_reduce(disparity_left, (4,4), np.max)
        file_name = os.path.join(disp_dir, "P2%06d.png" % i)
        cv2.imwrite(file_name, disparity_left)

        disparity_right = skimage.measure.block_reduce(disparity_right, (4,4), np.max)
        file_name = os.path.join(disp_dir, "P3%06d.png" % i)
        cv2.imwrite(file_name, disparity_left)

    print(f'{data_split} split finished precomputing disparity')
    print(f'Saved the results in {disp_dir}')


def main():
    cfg = load_config(args.config)
    data_dir = args.data_dir

    # no need for image, could be modified for extended use
    output_dict = {
        'calib': True,
        'image': True,
        'image_3' : True,
        'label': False,
        'velodyne': args.use_point_cloud,
    }

    train_names, _ = load_names(data_dir)
    print(f'Loaded {len(train_names)} names')

    data_root_dir = os.path.join(data_dir, 'training')
    compute_dispairity_for_split(
        cfg,
        train_names,
        data_root_dir,
        output_dict,
        'training',
        args.use_point_cloud
    )

    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data-dir', type=str, help='Data root directory')
    parser.add_argument('--use_point_cloud', action='store_true',
                        help='Use point cloud')
    args = parser.parse_args()
    print(args)
    main()