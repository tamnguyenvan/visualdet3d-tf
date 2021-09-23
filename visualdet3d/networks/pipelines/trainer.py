import numpy as np

import tensorflow as tf
from visualdet3d.utils import compound_annotation
from visualdet3d.networks.utils.registry import PIPELINE_DICT


@PIPELINE_DICT.register_module
def train_stereo_detection(data,
                           model,
                           optimizer,
                           cfg=None):
    """
    """
    left_images, right_images, P2, P3, labels, bbox2d, bbox_3d, disparity = data
    max_length = np.max([len(label) for label in labels])
    if max_length == 0:
        return

    annotation = compound_annotation(labels, max_length, bbox2d, bbox_3d, cfg.obj_types)  #np.arraym, [batch, max_length, 4 + 1 + 7]
    import pdb;pdb.set_trace()
    with tf.GradientTape() as tape:
        cls_loss, reg_loss, loss_dict = model(
            [left_images, right_images, annotation, P2, P3, disparity],
            training=True
        )

        cls_loss = tf.reduce_mean(cls_loss)
        reg_loss = tf.reduce_mean(reg_loss)
        
        loss = cls_loss + reg_loss
    
    # grads = tape.gradient(loss, model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss