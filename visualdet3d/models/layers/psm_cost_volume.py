import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from visualdet3d.models.layers.common import conv3d_3x3


class CostVolume(layers.Layer):
    def __init__(self, max_disp=192, downsample_scale=4, input_features=1024, psm_features=64):
        super(CostVolume, self).__init__()
        self.max_disp = max_disp
        self.downsample_scale = downsample_scale
        self.depth_channel = int(self.max_disp / self.downsample_scale)
        self.downsample = keras.Sequential([
            layers.Conv2D(psm_features, kernel_regularizer=1),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])

        self.conv3d = keras.Sequential([
            conv3d_3x3(psm_features, padding=1),
            layers.BatchNormalization(),
            layers.ReLU(),
            conv3d_3x3(psm_features, padding=1),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])
    
    def call(self, left_features, right_features):
        batch_size, _, w, h = left_features.shape
        left_features = self.downsample(left_features)
        right_features = self.downsample(right_features)
        
        cost = tf.Variable(tf.constant([
            left_features.shape[0],
            left_features.shape[1] * 2,
            self.depth_channel,
            left_features.shape[2],
            left_features.shape[3]
        ]))

        for i in range(self.depth_channel):
            if i > 0:
                cost[:, :left_features.shape[1], i, :, i:] = left_features[:, :, :, i:]
                cost[:, left_features.shape[1]:, i, :, i:] = right_features[:, :, :, :-i]
            else:
                cost[:, :left_features.shape[1], i, :, i:] = left_features
                cost[:, left_features.shape[1]:, i, :, i:] = right_features
        
        cost = self.conv3d(cost)
        cost = cost.reshape(batch_size, -1, w, h)
        return cost


class PSMCosineLayer(layers.Layer):
    def __init__(self, max_disp=192, downsample_scale=4):
        super(PSMCosineLayer, self).__init__()
        self.max_disp = max_disp
        self.downsample_scale = downsample_scale
        self.depth_channel = int(self.max_disp / self.downsample_scale)

    
    def call(self, left_features, right_features):
        cost = tf.Variable(tf.constant([
            left_features.shape[0],
            self.depth_channel,
            left_features.shape[2],
            left_features.shape[3],
        ]))

        for i in range(self.depth_channel):
            if i > 0:
                cost[:, i, :, i:] = (left_features[:, :, :, i:] * right_features[:, :, :, :-i]).mean(axis=1)
            else:
                cost[:, i, :, :] = (left_features * right_features).mean(axis=1)
        return cost


class DoublePSMCosineLayer(PSMCosineLayer):
    def __init__(self, max_disp=192, downsample_scale=4):
        super(DoublePSMCosineLayer, self).__init__(max_disp=max_disp, downsample_scale=downsample_scale)
        self.depth_channel = self.depth_channel
    
    def call(self, left_features, right_features):
        b, c, h, w = left_features.shape
        # TODO:
        return 0