"""
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from visualdet3d.models.layers.ghost import ResGhostLayer
from visualdet3d.models.backbones.resnet import BasicBlock, ResNet
from visualdet3d.models.layers.psm_cost_volume import PSMCosineLayer
from visualdet3d.models.layers.common import conv1x1, conv3x3


class CostVolumePyramid(layers.Layer):
    def __init__(self, depth_channel_4, depth_channel_8, depth_channel_16):
        super(CostVolumePyramid, self).__init__()
        self.depth_channel_4 = depth_channel_4
        self.depth_channel_8 = depth_channel_8
        self.depth_channel_16 = depth_channel_16

        input_features = depth_channel_4
        self.four_to_eight = keras.Sequential([
            ResGhostLayer(input_features, 3 * input_features, 3, ratio=3),
            layers.AveragePooling2D(2),
            BasicBlock(3 * input_features),
        ])

        input_features = 3 * input_features + depth_channel_8
        self.eight_to_sixteen = keras.Sequential([
            ResGhostLayer(input_features, 3 * input_features, 3, ratio=3),
            layers.AveragePooling2D(2),
            BasicBlock(3 * input_features),
        ])

        input_features = 3 * input_features + depth_channel_16
        self.depth_reason = keras.Sequential([
            ResGhostLayer(input_features, 3 * input_features, 3, ratio=3),
            BasicBlock(3 * input_features),
        ])

        self.output_channels = 3 * input_features
        self.depth_output = keras.Sequential([
            layers.UpSampling2D(size=2, interpolation='bilinear'),
            conv3x3(int(self.output_channels / 2), padding=1),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.UpSampling2D(size=2, interpolation='bilinear'),
            conv3x3(int(self.output_channels / 4), padding=1),
            layers.BatchNormalization(),
            layers.ReLU(),
            conv1x1(96),
        ])
    
    def call(self, psv_volume_4, psv_volume_8, psv_volume_16):
        psv_4_8 = self.four_to_eight(psv_volume_4)
        psv_volume_8 = tf.concat([psv_4_8, psv_volume_8], axis=1)
        psv_8_16 = self.eight_to_sixteen(psv_volume_8)
        psv_volume_16 = tf.concat([psv_8_16, psv_volume_16], axis=1)
        psv_16 = self.depth_reason(psv_volume_16)
        if self.training:
            return psv_16, self.depth_output(psv_16)
        
        return psv_16, tf.zeros([psv_volume_4.shape[0], 1, psv_volume_4.shape[2], psv_volume_4.shape[3]])


class StereoMerging(layers.Layer):
    def __init__(self, base_features):
        super(StereoMerging, self).__init__()
        self.cost_volume_0 = PSMCosineLayer(downsample_scale=4, max_disp=96)
        psv_depth_0 = self.cost_volume_0.depth_channel
        
        self.cost_volume_1 = PSMCosineLayer(downsample_scale=8, max_disp=192)
        psv_depth_1 = self.cost_volume_1.depth_channel

        self.cost_volume_2 = PSMCosineLayer(downsample_scale=16, max_disp=192)
        psv_depth_2 = self.cost_volume_2.depth_channel

        self.depth_reasoning = CostVolumePyramid(psv_depth_0, psv_depth_1, psv_depth_2)
        self.final_channel = self.depth_reasoning.output_channels + base_features * 4

    def call(self, left_x, right_x):
        psv_volume_0 = self.cost_volume_0(left_x[0], right_x[0])
        psv_volume_1 = self.cost_volume_1(left_x[1], right_x[2])
        psv_volume_2 = self.cost_volume_2(left_x[1], right_x[2])
        psv_features, depth_output = self.depth_reasoning(psv_volume_0, psv_volume_1, psv_volume_2)
        features = tf.concat([left_x[2], psv_features], axis=1)
        return features, depth_output


class YOLOStereo3DCore(keras.Model):
    """
        Inference Structure of YoloStereo3D
        Similar to YoloMono3D,
        Left and Right image are fed into the backbone in batch. So they will affect each other with BatchNorm2d.
    """
    def __init__(self, backbone_arguments):
        super(YOLOStereo3DCore, self).__init__()
        self.backbone = ResNet(**backbone_arguments)

        base_features = 256 if backbone_arguments['depth'] > 34 else 64
        self.neck = StereoMerging(base_features)
    
    def call(self, images):
        batch_size = images.shape[0]
        left_images = images[:, 0:3, :, :]
        right_images = images[:, 3:, :, :]
        
        images = tf.concat([left_images, right_images], axis=0)
        features = self.backbone(images)

        left_features = [feature[0:batch_size] for feature in features]
        right_features = [feature[batch_size:] for feature in features]

        features, depth_output = self.neck(left_features, right_features)
        output_dict = dict(features=features, depth_output=depth_output)
        return output_dict