import os
import argparse

import numpy as np
from visualdet3d.data.kitti.stereo_dataset import KittiStereoDataset
from visualdet3d.models.detectors import get_detector
from visualdet3d.optim.optimizers import get_optimizer
from visualdet3d.optim.schedulers import get_scheduler
from configs import load_config


import tensorflow as tf
from visualdet3d.utils import compound_annotation


def train_stereo_detection(data,
                           model,
                           optimizer,
                           global_step=None,
                           epoch_num=None,
                           cfg=None):
    """
    """
    left_images, right_images, P2, P3, labels, bbox2d, bbox_3d, disparity = data
    max_length = np.max([len(label) for label in labels])
    if max_length == 0:
        return

    annotation = compound_annotation(labels, max_length, bbox2d, bbox_3d, cfg.classes)  #np.arraym, [batch, max_length, 4 + 1 + 7]
    cls_loss, reg_loss, loss_dict = model(
        [left_images, right_images, annotation, P2, P3, disparity]
    )

    cls_loss = tf.reduce_mean(cls_loss)
    reg_loss = tf.reduce_mean(reg_loss)
    
    loss = cls_loss + reg_loss
    return loss


def main():
    # Load config
    cfg = load_config(args.config)

    # Create dataset
    train_loader = KittiStereoDataset(cfg, split='training')
    val_loader = KittiStereoDataset(cfg, split='validation')

    # Build model
    model = get_detector(cfg)

    # Build optimizer and scheduler
    optimizer = get_optimizer(cfg)
    lr_scheduler = get_scheduler(cfg)

    # Train
    for epoch in range(cfg.trainer.max_epochs):
        for idx, data in enumerate(train_loader):
            loss = train_stereo_detection(data, model, optimizer, cfg=cfg)
            print('Loss:', loss.numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file')
    args = parser.parse_args()
    main()