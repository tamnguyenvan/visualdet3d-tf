"""
"""
import os
import argparse
import pickle
from copy import deepcopy

import tqdm
import numpy as np
import tensorflow as tf

import context
from visualdet3d.models.heads.anchors import Anchors
from visualdet3d.data.kitti.preprocessing import KittiData
from visualdet3d.models.utils import calc_iou
from visualdet3d.utils import Timer
from visualdet3d.data.pipeline.transforms import get_transform
from configs import load_config


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


def read_one_split(cfg,
                   index_names,
                   data_root_dir,
                   output_dict,
                   data_split='training',
                   time_display_inter=100):
    """
    """
    # Prepare output directories
    save_dir = os.path.join(cfg.path.preprocessed_path, data_split)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if data_split == 'training':
        disp_dir = os.path.join(save_dir, 'disp')
        if not os.path.isdir(disp_dir):
            os.mkdir(disp_dir)

    N = len(index_names)
    frames = [None] * N
    print(f'Reading {data_split} data')
    timer = Timer()

    anchor_prior = getattr(cfg, 'anchor_prior', True)

    total_objects = [0 for _ in range(len(cfg.classes))]
    total_usable_objects = [0 for _ in range(len(cfg.classes))]
    if anchor_prior:
        anchor_manager = Anchors(cfg.path.preprocessed_path,
                                 read_config_file=False,
                                 data_format='channels_first',
                                 **cfg.detector.head.anchors_cfg)
        preprocess = get_transform(cfg.data.augmentation.test)
        total_objects = [0 for _ in range(len(cfg.classes))]
        total_usable_objects = [0 for _ in range(len(cfg.classes))]
        
        len_scale = len(anchor_manager.scales)
        len_ratios = len(anchor_manager.ratios)
        len_level = len(anchor_manager.pyramid_levels)

        examine = np.zeros([len(cfg.classes), len_level * len_scale, len_ratios])
        sums = np.zeros([len(cfg.classes), len_level * len_scale, len_ratios, 3]) 
        squared = np.zeros([len(cfg.classes), len_level * len_scale, len_ratios, 3], dtype=np.float64)

        uniform_sum_each_type = np.zeros((len(cfg.classes), 6), dtype=np.float64) #[z, sin2a, cos2a, w, h, l]
        uniform_square_each_type = np.zeros((len(cfg.classes), 6), dtype=np.float64)

    print('Loading data...')
    for i, index_name in tqdm.tqdm(enumerate(index_names)):
        # read data with dataloader api
        data_frame = KittiData(data_root_dir, index_name, output_dict)
        calib, image, label, velo = data_frame.read_data()

        # store the list of kittiObjet and kittiCalib
        max_occlusion = getattr(cfg.data, 'max_occlusion', 2)
        min_z = getattr(cfg.data, 'min_z', 3)
        if data_split == 'training':
            data_frame.label = [obj for obj in label.data if obj.type in cfg.classes and obj.occluded < max_occlusion and obj.z > min_z]
            
            if anchor_prior:
                for j in range(len(cfg.classes)):
                    total_objects[j] += len([obj for obj in data_frame.label if obj.type==cfg.classes[j]])
                    data = np.array(
                        [
                            [obj.z, np.sin(2 * obj.alpha), np.cos(2 * obj.alpha), obj.w, obj.h, obj.l] 
                                for obj in data_frame.label if obj.type==cfg.classes[j]
                        ]
                    ) #[N, 6]
                    if data.any():
                        uniform_sum_each_type[j, :] += np.sum(data, axis=0)
                        uniform_square_each_type[j, :] += np.sum(data ** 2, axis=0)
        else:
            data_frame.label = [obj for obj in label.data if obj.type in cfg.classes]
        data_frame.calib = calib
        
        if data_split == 'training' and anchor_prior:
            original_image = image.copy()
            baseline = (calib.P2[0, 3] - calib.P3[0, 3]) / calib.P2[0, 0]
            image, P2, label = preprocess(original_image, p2=deepcopy(calib.P2), labels=deepcopy(data_frame.label))
            _,  P3 = preprocess(original_image, p2=deepcopy(calib.P3))

            ## Computing statistic for positive anchors
            if len(data_frame.label) > 0:
                anchors, _ = anchor_manager(
                    image[np.newaxis].transpose([0, 3, 1, 2]),
                    tf.reshape(tf.convert_to_tensor(P2), ([-1, 3, 4]))
                )

                for j in range(len(cfg.classes)):
                    bbox2d = tf.convert_to_tensor(
                        [[obj.bbox_l, obj.bbox_t, obj.bbox_r, obj.bbox_b] for obj in label if obj.type == cfg.classes[j]])
                    if len(bbox2d) < 1:
                        continue
                    bbox3d = tf.constant([
                        [obj.x, obj.y, obj.z, np.sin(2 * obj.alpha), np.cos(2 * obj.alpha)]
                            for obj in label if obj.type == cfg.classes[j]
                    ])
                    
                    usable_anchors = anchors[0]

                    iou = calc_iou(usable_anchors, bbox2d) #[N, K]
                    # IoU_max, IoU_argmax = torch.max(IoUs, dim=0)
                    iou_max = tf.reduce_max(iou, axis=0)
                    # iou_max_anchor, IoU_argmax_anchor = torch.max(iou, dim=1)
                    iou_max_anchor = tf.reduce_max(iou, axis=1)
                    iou_argmax_anchor = tf.argmax(iou, axis=1)

                    num_usable_object = int(tf.squeeze(
                        tf.reduce_sum(
                            tf.cast(iou_max > cfg.detector.head.loss_cfg.fg_iou_threshold, tf.int32)
                        )
                    ).numpy())
                    total_usable_objects[j] += num_usable_object

                    positive_anchors_mask = iou_max_anchor > cfg.detector.head.loss_cfg.fg_iou_threshold
                    # positive_ground_truth_3d = bbox3d[iou_argmax_anchor[positive_anchors_mask]].numpy()
                    positive_ground_truth_3d = tf.gather_nd(
                        bbox3d,
                        indices=tf.reshape(iou_argmax_anchor[positive_anchors_mask], (-1, 1))).numpy()

                    used_anchors = usable_anchors[positive_anchors_mask].numpy() #[x1, y1, x2, y2]

                    sizes_int, ratio_int = anchor_manager.anchors_to_indexes(used_anchors)
                    for k in range(len(sizes_int)):
                        examine[j, sizes_int[k], ratio_int[k]] += 1
                        sums[j, sizes_int[k], ratio_int[k]] += positive_ground_truth_3d[k, 2:5]
                        squared[j, sizes_int[k], ratio_int[k]] += positive_ground_truth_3d[k, 2:5] ** 2

        frames[i] = data_frame

        if (i+1) % time_display_inter == 0:
            avg_time = timer.compute_avg_time(i+1)
            eta = timer.compute_eta(i+1, N)
            print("{} iter:{}/{}, avg-time:{}, eta:{}, total_objs:{}, usable_objs:{}".format(
                data_split, i+1, N, avg_time, eta, total_objects, total_usable_objects), end='\r')

    save_dir = os.path.join(cfg.path.preprocessed_path, data_split)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if data_split == 'training' and anchor_prior:
        
        for j in range(len(cfg.classes)):
            global_mean = uniform_sum_each_type[j] / total_objects[j]
            global_var = np.sqrt(uniform_square_each_type[j] / total_objects[j] - global_mean ** 2)

            avg = sums[j] / (examine[j][:, :, np.newaxis] + 1e-8)
            EX_2 = squared[j] / (examine[j][:, :, np.newaxis] + 1e-8)
            std = np.sqrt(EX_2 - avg ** 2)

            avg[examine[j] < 10, :] = -100  # with such negative mean Z, anchors/losses will filter them out
            std[examine[j] < 10, :] = 1e10
            avg[np.isnan(std)] = -100
            std[np.isnan(std)] = 1e10
            avg[std < 1e-3] = -100
            std[std < 1e-3] = 1e10

            whl_avg = np.ones([avg.shape[0], avg.shape[1], 3]) * global_mean[3:6]
            whl_std = np.ones([avg.shape[0], avg.shape[1], 3]) * global_var[3:6]

            avg = np.concatenate([avg, whl_avg], axis=2)
            std = np.concatenate([std, whl_std], axis=2)

            npy_file = os.path.join(save_dir,'anchor_mean_{}.npy'.format(cfg.classes[j]))
            np.save(npy_file, avg)
            std_file = os.path.join(save_dir,'anchor_std_{}.npy'.format(cfg.classes[j]))
            np.save(std_file, std)
    pkl_file = os.path.join(save_dir,'imdb.pkl')
    pickle.dump(frames, open(pkl_file, 'wb'))
    print("{} split finished precomputing".format(data_split))


def main():
    cfg = load_config(args.config)
    data_dir = args.data_dir
    
    time_display_inter = 100

    # No need for image, could be modified for extended use
    output_dict = {
        'calib': True,
        'image': True,
        'label': True,
        'velodyne': False,
    }

    train_names, val_names = load_names(data_dir)
    print(f'Loaded {len(train_names)} training names')
    print(f'Loaded {len(val_names)} validation names')

    data_root_dir = os.path.join(data_dir, 'training')
    read_one_split(cfg, train_names, data_root_dir, output_dict, 'training', time_display_inter)
    output_dict = {
        'calib': True,
        'image': False,
        'label': True,
        'velodyne': False,
    }
    read_one_split(cfg, val_names, data_root_dir, output_dict, 'validation', time_display_inter)

    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data-dir', type=str, help='Data root directory')
    parser.add_argument('--use_point_cloud', action='store_true',
                        help='Use point cloud')
    args = parser.parse_args()
    print(args)
    main()
