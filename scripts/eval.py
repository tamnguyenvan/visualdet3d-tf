import os
import argparse

import tensorflow as tf
from visualdet3d.data.kitti.stereo_dataset import KittiStereoDataset, KittiStereoTestDataset
from visualdet3d.models.detectors import get_detector
from visualdet3d.evaluator import evaluate_kitti_obj
from configs import load_config


def main():
    # Read Config
    cfg = load_config(args.cfg)
    
    # Set up dataset and dataloader
    split = args.split
    checkpoint_path = args.ckpt
    if split == 'training':
        dataset = KittiStereoDataset(cfg, split)
    elif split == 'test':
        dataset = KittiStereoTestDataset(cfg, split)
        cfg.is_running_test_set = True
    else:
        dataset = KittiStereoDataset(cfg, split)

    # Create the model
    detector = get_detector(cfg)
    detector.load_weights()

    # Run evaluation
    evaluate_kitti_obj(cfg, detector, dataset, None, 0, result_path_split=split)
    print('finish')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--split', type=str, default=['validation'],
                        help='Split to test')
    parser.add_argument('--ckpt', type=str, help='Path to checkpoint/saved model')
    args = parser.parse_args()
    main()