import os
import argparse

import numpy as np
import tensorflow as tf
from visualdet3d.data.kitti.stereo_dataset import KittiStereoDataset
from visualdet3d.models.detectors import get_detector
from visualdet3d.optim.optimizers import get_optimizer
from visualdet3d.optim.schedulers import get_scheduler
from visualdet3d.models.pipelines.trainer import train_stereo_detection
from visualdet3d.models.utils import get_num_parameters
from configs import load_config



def main():
    # Load config
    cfg = load_config(args.config)

    # Create dataset
    train_loader = KittiStereoDataset(cfg, split='training')
    val_loader = KittiStereoDataset(cfg, split='validation')

    # Build model
    model = get_detector(cfg)
    # num_params = get_num_parameters(model)
    # print(f'Number of trainable parameters: {num_params//1e6}M')

    # Build optimizer and scheduler
    lr_scheduler = get_scheduler(cfg)
    optimizer = get_optimizer(lr_scheduler, cfg)

    # Train
    for epoch in range(cfg.trainer.max_epochs):
        for idx, data in enumerate(train_loader):
            loss = train_stereo_detection(data, model, optimizer, cfg=cfg)
            print(f'[Epoch {epoch+1:03d} iter {idx+1:04d}] Loss: {loss.numpy():.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file')
    args = parser.parse_args()
    main()