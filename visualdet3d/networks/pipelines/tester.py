from typing import Tuple, List

import tensorflow as tf
from easydict import EasyDict
from visualdet3d.networks.utils.registry import PIPELINE_DICT


@PIPELINE_DICT.register_module
def test_stereo_detection(data,
                          module: tf.keras.Model,
                          cfg: EasyDict=None) -> Tuple[tf.Tensor, tf.Tensor, List[str]]:
    """
    """
    left_images, right_images, P2, P3 = data[0], data[1], data[2], data[3]

    scores, bbox, obj_index = module([
        left_images,
        right_images,
        tf.constant(P2),
        tf.constant(P3)
    ], training=False)
    scores = scores.numpy()
    bbox = bbox.numpy()
    obj_index = obj_index.numpy()
    obj_types = [cfg.obj_types[int(i)] for i in obj_index]

    return scores, bbox, obj_types