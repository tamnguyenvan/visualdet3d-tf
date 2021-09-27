import os
import argparse

import tensorflow as tf

import context
from visualdet3d.networks.utils.registry import DETECTOR_DICT, DATASET_DICT, PIPELINE_DICT
from configs import load_cfg


def main():
    # Read Config
    cfg = load_cfg(args.config)
    
    # Force GPU selection in command line
    cfg.trainer.gpu = 0

    # Set up dataset and dataloader
    split = args.split
    is_test_train = split == 'training'
    if split == 'training':
        dataset_name = cfg.data.train_dataset
    elif split == 'test':
        dataset_name = cfg.data.test_dataset
        cfg.is_running_test_set = True
    else:
        dataset_name = cfg.data.val_dataset
    dataset = DATASET_DICT[dataset_name](cfg, split)

    # Create the model
    detector = DETECTOR_DICT[cfg.detector.name](cfg.detector)
    detector.load_weights(args.ckpt).expect_partial()

    if 'evaluate_func' in cfg.trainer:
        evaluate_detection = PIPELINE_DICT[cfg.trainer.evaluate_func]
        print("Found evaluate function")
    else:
        raise KeyError("evluate_func not found in Config")

    # Run evaluation
    evaluate_detection(cfg, detector, dataset, split, result_path_split=split)
    print('finish')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--split', type=str, default='validation',
                        help='Split to test')
    parser.add_argument('--gpu', type=str, default='0', help='GPUs')
    parser.add_argument('--logdir', type=str, default='logs',
                        help='Tensorboard logs dir')
    parser.add_argument('--ckpt', type=str, help='Path to checkpoint/saved model')
    args = parser.parse_args()
    main()