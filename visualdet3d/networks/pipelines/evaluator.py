import os
import shutil
from typing import Sized, Sequence

import cv2
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict

import tensorflow as tf
from visualdet3d.evaluator.kitti.evaluate import evaluate
from visualdet3d.evaluator.kitti_depth_prediction.evaluate_depth import evaluate_depth
from visualdet3d.networks.utils import BBox3dProjector, BackProjection
from visualdet3d.data.kitti.utils import write_result_to_file
from visualdet3d.networks.utils.registry import PIPELINE_DICT


@PIPELINE_DICT.register_module
def evaluate_kitti_depth(cfg: edict,
                         model: tf.keras.Model,
                         dataset_val: Sequence,
                         result_path_split='validation'):
    """
    """
    result_path = os.path.join(cfg.path.preprocessed_path, result_path_split, 'data')
    if os.path.isdir(result_path):
        shutil.rmtree(result_path)
        print('Clean up the recorder directory of {result_path}')
    os.mkdir(result_path)

    print('Rebuild {}'.format(result_path))
    for index in tqdm(range(len(dataset_val))):
        data = dataset_val[index]
        collated_data = dataset_val.collate_fn([data])
        image, K = collated_data
        return_dict = model(
                [image.cuda().float(), image.new(K)]
            )
        depth = return_dict['target'][0, 0]
        depth_uint16 = (depth * 256).cpu().numpy().astype(np.uint16)
        w, h = data['original_shape'][1], data['original_shape'][0]
        height_to_pad = h - depth_uint16.shape[0]
        depth_uint16 = np.pad(depth_uint16, [(height_to_pad, 0), (0, 0)], mode='edge')
        depth_uint16 = cv2.resize(depth_uint16, (w, h))
        depth_uint16[depth_uint16 == 0] = 1 
        image_name = '%010d.png' % index
        cv2.imwrite(os.path.join(result_path, image_name), depth_uint16)

    if 'is_running_test_set' in cfg and cfg['is_running_test_set']:
        print('Finish evaluation.')
        return
    result_texts = evaluate_depth(
        label_path = os.path.join(cfg.path.validation_path, 'groundtruth_depth'),
        result_path = result_path
    )
    # for index, result_text in enumerate(result_texts):
    #     if writer is not None:
    #         writer.add_text('validation result {}'.format(index), result_text.replace(' ', '&nbsp;').replace('\n', '  \n'), epoch_num + 1)
    #     print(result_text, end='')
    # print()


@PIPELINE_DICT.register_module
def evaluate_kitti_obj(cfg: edict, 
                       model: tf.keras.Model,
                       dataset_val: Sized,
                       split='validation',
                       epoch_num=0,
                       writer=None,
                       result_path_split='validation'):
    """
    """
    result_path = os.path.join(cfg.path.preprocessed_path, result_path_split, 'data')
    if os.path.isdir(result_path):
        os.system('rm -r {}'.format(result_path))
        print('Clean up the recorder directory of {}'.format(result_path))
    os.mkdir(result_path)
    print('Rebuild {}'.format(result_path))
    test_func = PIPELINE_DICT[cfg.trainer.test_func]
    projector = BBox3dProjector()
    backprojector = BackProjection()
    batch_size = cfg.data.batch_size
    for index in tqdm(range(len(dataset_val))):
        test_batch(cfg, split, index, batch_size, dataset_val,
                   model, test_func, backprojector, projector, result_path)
    if 'is_running_test_set' in cfg and cfg['is_running_test_set']:
        print('Finish evaluation.')
        return
    result_texts = evaluate(
        label_path=os.path.join(cfg.path.data_path, 'label_2'),
        result_path=result_path,
        label_split_file=cfg.data.val_split_file,
        current_classes=[i for i in range(len(cfg.classes))],
        gpu=0
    )
    for class_index, result_text in enumerate(result_texts):
        if writer is not None:
            writer.add_text('validation result {}'.format(class_index), result_text.replace(' ', '&nbsp;').replace('\n', '  \n'), epoch_num + 1)
        print(result_text)


def test_batch(cfg,
               split,
               index,
               batch_size,
               dataset,
               model,
               test_func,
               backprojector: BackProjection,
               projector: BBox3dProjector,
               result_path):
    """
    """
    if split == 'validation':
        if len(dataset[index]) == 7:
            left_images, right_images, P2, P3, labels, bbox2ds, bbox3ds = dataset[index]
            disparities = None
        else:
            left_images, right_images, P2, P3, labels, bbox2ds, bbox3ds, disparities = dataset[index]
        N = len(left_images)
        for i in range(N):
            data = {
                'calib': [P2[i], P3[i]],
                'image': [left_images[i], right_images[i]],
                'label': labels[i], 
                'bbox2d': bbox2ds[i], #[N, 4] [x1, y1, x2, y2]
                'bbox3d': bbox3ds[i],
                'original_shape': left_images[i].shape,
                'disparity': disparities if disparities is None else disparities[i],
                'original_P': P2[i]
            }
            test_one(cfg,
                    data,
                    index*batch_size+i,
                    dataset,
                    model,
                    test_func,
                    backprojector,
                    projector,
                    result_path)
    elif split == 'test':
        batch_data = dataset[index]
        for i, data in enumerate(batch_data):
            test_one(cfg,
                    data,
                    index*batch_size+i,
                    dataset,
                    model,
                    test_func,
                    backprojector,
                    projector,
                    result_path)
    else:
        raise ValueError(f'Not support split {split}')


def test_one(cfg,
             data,
             index,
             dataset,
             model,
             test_func,
             backprojector: BackProjection,
             projector: BBox3dProjector,
             result_path):
    """
    """
    if isinstance(data['calib'], list):
        P2 = data['calib'][0]
    else:
        P2 = data['calib']
    original_height = data['original_shape'][0]
    collated_data = dataset.collate_fn([data])
    height = collated_data[0].shape[2]

    scores, bbox, obj_names = test_func(collated_data, model, cfg=cfg)
    bbox_2d = bbox[:, 0:4]
    if bbox.shape[1] > 4: # run 3D
        bbox_3d_state = bbox[:, 4:] #[cx,cy,z,w,h,l,alpha, bot, top]
        bbox_3d_state_3d = backprojector(bbox_3d_state, P2) #[x, y, z, w,h ,l, alpha, bot, top]

        _, _, thetas = projector(bbox_3d_state_3d, bbox_3d_state_3d.new(P2))

        original_P = data['original_P']
        scale_x = original_P[0, 0] / P2[0, 0]
        scale_y = original_P[1, 1] / P2[1, 1]
        
        shift_left = original_P[0, 2] / scale_x - P2[0, 2]
        shift_top  = original_P[1, 2] / scale_y - P2[1, 2]
        bbox_2d[:, 0:4:2] += shift_left
        bbox_2d[:, 1:4:2] += shift_top

        bbox_2d[:, 0:4:2] *= scale_x
        bbox_2d[:, 1:4:2] *= scale_y

        write_result_to_file(result_path, index, scores, bbox_2d, bbox_3d_state_3d, thetas, obj_names)
    else:
        if 'crop_top' in cfg.data.augmentation and cfg.data.augmentation.crop_top is not None:
            crop_top = cfg.data.augmentation.crop_top
        elif 'crop_top_height' in cfg.data.augmentation and cfg.data.augmentation.crop_top_height is not None:
            if cfg.data.augmentation.crop_top_height >= original_height:
                crop_top = 0
            else:
                crop_top = original_height - cfg.data.augmentation.crop_top_height

        scale_2d = (original_height - crop_top) / height
        bbox_2d[:, 0:4] *= scale_2d
        bbox_2d[:, 1:4:2] += cfg.data.augmentation.crop_top
        if isinstance(scores, tf.Tensor):
            scores = scores.numpy()
        write_result_to_file(result_path, index, scores, bbox_2d, obj_types=obj_names)