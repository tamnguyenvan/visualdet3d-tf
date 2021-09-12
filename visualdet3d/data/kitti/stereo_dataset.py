import os
import math
import pickle
from collections import OrderedDict
from copy import deepcopy
from typing import List

import numpy as np
import cv2
import tensorflow as tf

from visualdet3d.data.pipeline.transforms import get_transform
from visualdet3d.data.kitti.preprocessing import KittiObj
from visualdet3d.models.utils import BBox3dProjector


class KittiStereoDataset(tf.keras.utils.Sequence):
    """Kitti Stereo Dataset"""
    def __init__(self, cfg, split='training'):
        super(KittiStereoDataset, self).__init__()

        preprocessed_path   = cfg.path.preprocessed_path
        obj_types = cfg.classes
        is_train = (split == 'training')
        imdb_file_path = os.path.join(preprocessed_path, split, 'imdb.pkl')
        self.imdb = pickle.load(open(imdb_file_path, 'rb')) # list of kittiData
        self.output_dict = {
            'calib': True,
            'image': True,
            'image_3':True,
            'label': False,
            'velodyne': False
        }
        if is_train:
            self.transform = get_transform(cfg.data.augmentation.train)
        else:
            self.transform = get_transform(cfg.data.augmentation.test)
        self.cfg = cfg
        self.projector = BBox3dProjector()
        self.is_train = is_train
        self.obj_types = obj_types
        self.name_to_label = OrderedDict([(name, i) for i, name in enumerate(self.obj_types)])
        self.preprocessed_path = preprocessed_path

    def _reproject(self, P2: np.ndarray, transformed_label: List[KittiObj]):
        """
        """
        bbox3d_state = np.zeros([len(transformed_label), 7]) #[camera_x, camera_y, z, w, h, l, alpha]
        if len(transformed_label) > 0:
            #for obj in transformed_label:
            #    obj.alpha = theta2alpha_3d(obj.ry, obj.x, obj.z, P2)
            bbox3d_origin = tf.constant([
                [obj.x, obj.y - 0.5 * obj.h, obj.z, obj.w, obj.h, obj.l, obj.alpha]
                for obj in transformed_label], dtype=tf.float32)
            # try:
            abs_corner, homo_corner, _ = self.projector(bbox3d_origin, tf.constant(P2, dtype=bbox3d_origin.dtype))
            # except:
            #     print('\n',bbox3d_origin.shape, len(transformed_label), transformed_label, bbox3d_origin)

            for i, obj in enumerate(transformed_label):
                extended_center = np.array([obj.x, obj.y - 0.5 * obj.h, obj.z, 1])[:, np.newaxis] #[4, 1]
                extended_bottom = np.array([obj.x, obj.y, obj.z, 1])[:, np.newaxis] #[4, 1]
                image_center = (P2 @ extended_center)[:, 0] #[3]
                image_center[0:2] /= image_center[2]

                image_bottom = (P2 @ extended_bottom)[:, 0] #[3]
                image_bottom[0:2] /= image_bottom[2]
                
                bbox3d_state[i] = np.concatenate([image_center,
                                                 [obj.w, obj.h, obj.l, obj.alpha]]) #[7]

            max_xy = tf.reduce_max(homo_corner[:, :, 0:2], axis=1)  # [N,2]
            min_xy = tf.reduce_max(homo_corner[:, :, 0:2], axis=1)  # [N,2]

            result = tf.concat([min_xy, max_xy], axis=-1) #[:, 4]

            bbox2d = result.numpy()

            for i in range(len(transformed_label)):
                transformed_label[i].bbox_l = bbox2d[i, 0]
                transformed_label[i].bbox_t = bbox2d[i, 1]
                transformed_label[i].bbox_r = bbox2d[i, 2]
                transformed_label[i].bbox_b = bbox2d[i, 3]
        return transformed_label, bbox3d_state
    
    def __len__(self):
        return math.ceil(len(self.imdb) / self.cfg.data.batch_size)
    
    def _load_single_sample(self, idx):
        """
        """
        kitti_data = self.imdb[idx]

        # The calib and label has been preloaded to minimize the time in each indexing
        kitti_data.output_dict = self.output_dict
        calib, left_image, right_image, _, _ = kitti_data.read_data()
        calib.image_shape = left_image.shape
        label = []
        for obj in kitti_data.label:
            if obj.type in self.obj_types:
                label.append(obj)
        transformed_left_image, transformed_right_image, P2, P3, transformed_label = self.transform(
                left_image, right_image, deepcopy(calib.P2),deepcopy(calib.P3), deepcopy(label)
        )
        bbox3d_state = np.zeros([len(transformed_label), 7]) #[camera_x, camera_y, z, w, h, l, alpha]
        
        if len(transformed_label) > 0:
            transformed_label, bbox3d_state = self._reproject(P2, transformed_label)

        if self.is_train:
            if abs(P2[0, 3]) < abs(P3[0, 3]): # not mirrored or swaped, disparity should base on pointclouds projecting through P2
                disparity = cv2.imread(os.path.join(self.preprocessed_path, 'training', 'disp', "P2%06d.png" % idx), -1)
            else: # mirrored and swap, disparity should base on pointclouds projecting through P3, and also mirrored
                disparity = cv2.imread(os.path.join(self.preprocessed_path, 'training', 'disp', "P3%06d.png" % idx), -1)
                disparity = disparity[:, ::-1]
            disparity = disparity / 16.0
        else:
            disparity = None

        bbox2d = np.array([[obj.bbox_l, obj.bbox_t, obj.bbox_r, obj.bbox_b] for obj in transformed_label])
        
        output_dict = OrderedDict({
            'calib': [P2, P3],
            'image': [transformed_left_image, transformed_right_image],
            'label': [obj.type for obj in transformed_label], 
            'bbox2d': bbox2d,  #[N, 4] [x1, y1, x2, y2]
            'bbox3d': bbox3d_state,
            'original_shape': calib.image_shape,
            'disparity': disparity,
            'original_P':calib.P2.copy()
        })
        return output_dict
    
    def collate_fn(self, batch):
        # TODO: (B, C, H, W)?
        left_images = np.array([item['image'][0] for item in batch])  # [batch, H, W, 3]

        right_images = np.array([item['image'][1] for item in batch])  # [batch, H, W, 3]

        P2 = [item['calib'][0] for item in batch]
        P3 = [item['calib'][1] for item in batch]
        label = [item['label'] for item in batch]
        bbox2ds = [item['bbox2d'] for item in batch]
        bbox3ds = [item['bbox3d'] for item in batch]
        disparities = [item['disparity'] for item in batch]
        if disparities[0] is None:
            return (
                tf.constant(left_images, dtype=tf.float32),
                tf.constant(right_images, dtype=tf.float32),
                tf.constant(P2, dtype=tf.float32),
                tf.constant(P3, dtype=tf.float32),
                label,
                bbox2ds,
                bbox3ds
            )
        else:
            return (
                tf.constant(left_images, dtype=tf.float32),
                tf.constant(right_images, dtype=tf.float32),
                tf.constant(P2, dtype=tf.float32),
                tf.constant(P3, dtype=tf.float32),
                label,
                bbox2ds,
                bbox3ds,
                tf.constant(disparities, dtype=tf.float32),
            )
        

    def __getitem__(self, idx):
        batch_size = self.cfg.data.batch_size
        inputs_list = []
        for i in range(idx*batch_size, (idx+1)*batch_size):
            sample_input = self._load_single_sample(idx)
            inputs_list.append(sample_input)
        return self.collate_fn(inputs_list)


class KittiStereoTestDataset:
    def __init__(self, config, split='test'):
        pass