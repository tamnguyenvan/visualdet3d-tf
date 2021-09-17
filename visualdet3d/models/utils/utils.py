"""
"""
from typing import Union
# import torch
# import torch.nn as nn
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.backend import expand_dims, maximum
from visualdet3d.utils import alpha_to_theta_3d


def get_num_parameters(model):
    """Count number of trained parameters of the model"""
    num_parameters = 0
    for var in model.trainable_variables():
        num_parameters += tf.size(var)

    return num_parameters


def xyxy2xywh(box2d):
    """
        input   : [n, 4] [x1, y1, x2, y2]
        return  : [n, 4] [x, y, w, h]
        compatible with both pytorch and numpy
        a faster dedicated numpy implementation can be found at lib/fast_util/bbox2d.py
    """
    center_x = 0.5 * (box2d[:, 0] + box2d[:, 2])
    center_y = 0.5 * (box2d[:, 1] + box2d[:, 3])
    width_x  = box2d[:, 2] - box2d[:, 0]
    width_y  = box2d[:, 3] - box2d[:, 1]

    if isinstance(box2d, tf.Tensor):
        return tf.stack([center_x, center_y, width_x, width_y], axis=1)
    if isinstance(box2d, np.ndarray):
        return np.stack([center_x, center_y, width_x, width_y], axis=1)


def xywh2xyxy(box2d):
    """
        input   :  [n, 4] [x, y, w, h]
        return  :  [n, 4] [x1, y1, x2, y2]
        compatible with both pytorch and numpy
        a faster dedicated numpy implementation can be found at lib/fast_util/bbox2d.py
    """
    halfw = 0.5*box2d[:, 2]
    halfh = 0.5*box2d[:, 3]

    result_list = [
        box2d[:, 0] - halfw,
        box2d[:, 1] - halfh,
        box2d[:, 0] + halfw,
        box2d[:, 1] + halfh,
    ]
    if isinstance(box2d, tf.Tensor):
        return tf.stack(result_list, axis=1)
    if isinstance(box2d, np.ndarray):
        return np.stack(result_list, axis=1)


def cornerbbox2xyxy(corner_box):
    """Convert corner bbox(3D bbox corners projected on image) to 2D bounding boxes. Compatible with pytorch or numpy
    Args:
        corner_bbox(Union[Tensor, ndarray]) : [..., K, >=2] only the first two [x, y] are used.
    Return:
        bbox(Union[Tensor, ndarray])        : [..., 4] in the format of [x1, y1, x2, y2]
    """
    if isinstance(corner_box, tf.Tensor):
        max_xy = tf.max(corner_box[..., 0:2], axis=-2)  # [N,2]
        min_xy = tf.max(corner_box[..., 0:2], axis=-2)  # [N,2]

        result = tf.concat([min_xy, max_xy], axis=-1) #[:, 4]
        return result

    if isinstance(corner_box, np.ndarray):
        max_xy = corner_box[:, :, 0:2].max(axis=-2)
        min_xy = corner_box[:, :, 0:2].min(axis=-2)
        result = np.concatenate([max_xy, min_xy], axis=-1)
        return result

    else:
        raise NotImplementedError
        
def calc_iou(a, b):
    """Calculate Intersection over Union between `a` and `b`.
    
    Args
      a: tf.Tensor
        A Tensor shape of (N, 4)
      b: tf.Tensor
        A Tensor shape of (M, 4)
    
    Returns
      iou: tf.Tensor
        Intersection over Union matrix shape of (M, N)
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = tf.minimum(tf.expand_dims(a[:, 2], axis=1), b[:, 2]) - tf.maximum(tf.expand_dims(a[:, 0], axis=1), b[:, 0])
    ih = tf.minimum(tf.expand_dims(a[:, 3], axis=1), b[:, 3]) - tf.maximum(tf.expand_dims(a[:, 1], axis=1), b[:, 1])

    iw = tf.maximum(iw, 0.)
    ih = tf.maximum(ih, 0.)

    ua = tf.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih
    ua = tf.maximum(ua, 1e-8)

    intersection = iw * ih
    iou = intersection / ua
    return iou

class BBoxTransform(layers.Layer):
    """
        torch.nn Module that denormalized prediction from anchor box.
        Currently Compatible with 2D anchor_box  and 3D anchor box
        forward methods for bbox2d:
            input: 
                boxes:    (anchors of        [n1, n2, ..., 4])
                deltas:   (nn prediction of  [n1, n2, ..., 4])
        
        forward methods for bbox3d:
            input:
                boxes:    (anchors of        [n1, n2, ..., 4])  [x1, y1, x2, y2]
                deltas:   (nn prediction of  [n1, n2, ..., 9]) [x1, y1, x2, y2, cx, cy, z, s2a, c2a]
                anchors_mean_std: [types, N, 6, 2] including [z, s2a, c2a] mean and std for each anchors
                classes_index: [N] long index for types
            return:
                [N, 13]: [x1, y1, x2, y2, cx, cy, z, w, h, l, alpha] denormalized
    """
    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            self.mean = tf.convert_to_tensor(np.array([0, 0, 0, 0]).astype(np.float32))
        else:
            self.mean = mean
        if std is None:
            self.std = tf.convert_to_tensor(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))
        else:
            self.std = std

    def call(self, boxes, deltas,anchors_mean_std=None, label_index=None):

        widths  = boxes[..., 2] - boxes[..., 0]
        heights = boxes[..., 3] - boxes[..., 1]
        ctr_x   = boxes[..., 0] + 0.5 * widths
        ctr_y   = boxes[..., 1] + 0.5 * heights

        dx = deltas[..., 0] * self.std[0] + self.mean[0]
        dy = deltas[..., 1] * self.std[1] + self.mean[1]
        dw = deltas[..., 2] * self.std[2] + self.mean[2]
        dh = deltas[..., 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w     = tf.math.exp(dw) * widths
        pred_h     = tf.math.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h
        
        if not anchors_mean_std is None:
            one_hot_mask = tf.cast(tf.one_hot(label_index, anchors_mean_std.shape[1]), tf.bool)
            selected_mean_std = anchors_mean_std[one_hot_mask] #[N]
            mask = selected_mean_std[:, 0, 0] > 0
            
            cdx = deltas[..., 4] * self.std[0] + self.mean[0]
            cdy = deltas[..., 5] * self.std[0] + self.mean[0]
            pred_cx1 = ctr_x + cdx * widths
            pred_cy1 = ctr_y + cdy * heights
            pred_z   = deltas[...,6] * selected_mean_std[:, 0, 1] + selected_mean_std[:,0, 0]  #[N, 6]
            pred_sin = deltas[...,7] * selected_mean_std[:, 1, 1] + selected_mean_std[:,1, 0] 
            pred_cos = deltas[...,8] * selected_mean_std[:, 2, 1] + selected_mean_std[:,2, 0] 
            pred_alpha = tf.math.atan2(pred_sin, pred_cos) / 2.0

            pred_w = deltas[...,9]  * selected_mean_std[:, 3, 1] + selected_mean_std[:,3, 0]
            pred_h = deltas[...,10] * selected_mean_std[:,4, 1] + selected_mean_std[:,4, 0]
            pred_l = deltas[...,11] * selected_mean_std[:,5, 1] + selected_mean_std[:,5, 0]

            pred_boxes = tf.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2,
                                     pred_cx1, pred_cy1, pred_z,
                                     pred_w, pred_h, pred_l, pred_alpha], axis=1)
            return pred_boxes, mask

        pred_boxes = tf.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], axis=-1)

        return pred_boxes


class ClipBoxes(layers.Layer):

    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def call(self, boxes, img):

        batch_size, num_channels, height, width = img.shape
      
        boxes[:, 0] = tf.clip_by_value(boxes[:, 0], clip_value_min=0)
        boxes[:, 1] = tf.clip_by_value(boxes[:, 1], min=0)

        boxes[:, 2] = tf.clip_by_value(boxes[:, 2], clip_value_max=width)
        boxes[:, 3] = tf.clip_by_value(boxes[:, 3], clip_value_max=height)
        return boxes


class BBox3dProjector(layers.Layer):
    """
        forward methods
            input:
                unnormalize bbox_3d [N, 7] with  x, y, z, w, h, l, alpha
                tensor_p2: tensor of [3, 4]
            output:
                [N, 8, 3] with corner point in camera frame
                [N, 8, 3] with corner point in image frame
                [N, ] thetas
    """
    def __init__(self):
        super(BBox3dProjector, self).__init__()
        self.corner_matrix = tf.constant(
            [[-1, -1, -1],
            [ 1, -1, -1],
            [ 1,  1, -1],
            [ 1,  1,  1],
            [ 1, -1,  1],
            [-1, -1,  1],
            [-1,  1,  1],
            [-1,  1, -1]]
        , tf.float32)  # 8, 3

    def call(self, bbox_3d, tensor_p2):
        """
            input:
                unnormalize bbox_3d [N, 7] with  x, y, z, w, h, l, alpha
                tensor_p2: tensor of [3, 4]
            output:
                [N, 8, 3] with corner point in camera frame # 8 is determined by the shape of self.corner_matrix
                [N, 8, 3] with corner point in image frame
                [N, ] thetas
        """
        relative_eight_corners = 0.5 * self.corner_matrix * tf.expand_dims(bbox_3d[:, 3:6], 1)  # [N, 8, 3]
        # [batch, N, ]
        thetas = alpha_to_theta_3d(bbox_3d[..., 6], bbox_3d[..., 0], bbox_3d[..., 2], tensor_p2)
        _cos = tf.expand_dims(tf.math.cos(thetas), 1)  # [N, 1]
        _sin = tf.expand_dims(tf.math.sin(thetas), 1)  # [N, 1]
        rotated_corners_x, rotated_corners_z = (
            relative_eight_corners[:, :, 2] * _cos +
                relative_eight_corners[:, :, 0] * _sin,
        -relative_eight_corners[:, :, 2] * _sin +
            relative_eight_corners[:, :, 0] * _cos
        )  # relative_eight_corners == [N, 8, 3]

        rotated_corners = tf.stack([rotated_corners_x, relative_eight_corners[:,:,1], rotated_corners_z], axis=-1) #[N, 8, 3]
        abs_corners = rotated_corners + tf.expand_dims(bbox_3d[:, 0:3], 1)  # [N, 8, 3]
        camera_corners = tf.expand_dims(tf.concat([abs_corners,
            tf.ones([abs_corners.shape[0], self.corner_matrix.shape[0], 1], dtype=abs_corners.dtype)],
            axis=-1), 3)  # [N, 8, 4, 1]
        camera_coord = tf.squeeze(tf.matmul(tensor_p2, camera_corners), -1)  # [N, 8, 3]

        homo_coord = camera_coord / (camera_coord[:, :, 2:] + 1e-6) # [N, 8, 3]

        return abs_corners, homo_coord, thetas


class BackProjection(layers.Layer):
    """
        forward method:
            bbox3d: [N, 7] homo_x, homo_y, z, w, h, l, alpha
            p2: [3, 4]
            return [x3d, y3d, z, w, h, l, alpha]
    """
    def call(self, bbox3d, p2):
        """
            bbox3d: [N, 7] homo_x, homo_y, z, w, h, l, alpha
            p2: [3, 4]
            return [x3d, y3d, z, w, h, l, alpha]
        """
        fx = p2[0, 0]
        fy = p2[1, 1]
        cx = p2[0, 2]
        cy = p2[1, 2]
        tx = p2[0, 3]
        ty = p2[1, 3]

        z3d = bbox3d[:, 2:3] #[N, 1]
        x3d = (bbox3d[:,0:1] * z3d - cx * z3d - tx) / fx #[N, 1]
        y3d = (bbox3d[:,1:2] * z3d - cy * z3d - ty) / fy #[N, 1]
        return tf.concat([x3d, y3d, bbox3d[:, 2:]], axis=1)