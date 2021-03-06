import tensorflow as tf
from tensorflow import keras

from visualdet3d.networks.detectors.yolostereo3d_core import YOLOStereo3DCore
from visualdet3d.networks.heads.detection_3d_head import StereoHead
from visualdet3d.networks.heads import losses
from visualdet3d.networks.utils.registry import DETECTOR_DICT


@DETECTOR_DICT.register_module
class Stereo3D(keras.Model):
    def __init__(self, detector_cfg):
        super(Stereo3D, self).__init__()

        self.obj_types = detector_cfg.obj_types
        self.detector_cfg = detector_cfg
        self.build_head(detector_cfg)
        self.build_core(detector_cfg)
    
    def build_core(self, detector_cfg):
        self.core = YOLOStereo3DCore(detector_cfg.backbone, name='core')
    
    def build_head(self, detector_cfg):
        self.bbox_head = StereoHead(name='head', **(detector_cfg.head))
        self.disparity_loss = losses.DisparityLoss(maxdisp=96)
    
    def train_step(self, x):
        left_images, right_images, annotations, P2, P3, disparity = x
        output_dict = self.core(tf.concat([left_images, right_images], axis=-1))
        depth_output = output_dict['depth_output']

        cls_preds, reg_preds = self.bbox_head(
            dict(
                features=output_dict['features'],
                P2=P2,
                image=left_images
            )
        )

        anchors = self.bbox_head.get_anchor(left_images, P2, training=True)

        cls_loss, reg_loss, loss_dict = self.bbox_head.loss(
            cls_preds, reg_preds, anchors, annotations, P2)

        if tf.reduce_mean(reg_loss) > 0 and not disparity is None and not depth_output is None:
            disp_loss = 1.0 * self.disparity_loss(depth_output, disparity)
            loss_dict['disparity_loss'] = disp_loss
            reg_loss += disp_loss

            self.depth_output = depth_output
        else:
            loss_dict['disparity_loss'] = tf.zeros_like(reg_loss)
        return cls_loss, reg_loss, loss_dict
    
    def test_step(self, x):
        left_images, right_images, P2, P3 = x

        output_dict = self.core(tf.concat([left_images, right_images], axis=-1),
                                training=False)
        depth_output   = output_dict['depth_output']

        cls_preds, reg_preds = self.bbox_head(
            dict(
                features=output_dict['features'],
                P2=P2,
                image=left_images
            ),
            training=False
        )

        anchors = self.bbox_head.get_anchor(left_images, P2)
        scores, bboxes, cls_indexes = self.bbox_head.get_bboxes(
            cls_preds, reg_preds, anchors, P2, left_images)
        return scores, bboxes, cls_indexes
    
    def call(self, x, training=True):
        if training:
            return self.train_step(x)
        else:
            return self.test_step(x)