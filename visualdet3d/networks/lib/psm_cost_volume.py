import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from visualdet3d.networks.lib.common import conv1x1, conv3d_3x3


class CostVolume(layers.Layer):
    def __init__(self, max_disp=192, downsample_scale=4, psm_features=64):
        super(CostVolume, self).__init__()
        self.max_disp = max_disp
        self.downsample_scale = downsample_scale
        self.depth_channel = int(self.max_disp / self.downsample_scale)
        self.downsample = keras.Sequential([
            conv1x1(psm_features, padding=0),
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
        self.output_channel = psm_features * self.depth_channel
    
    def call(self, left_features, right_features):
        batch_size, w, h, _ = left_features.shape
        left_features = self.downsample(left_features)
        right_features = self.downsample(right_features)
        
        cost = tf.Variable(tf.zeros([
            left_features.shape[0],
            left_features.shape[1],
            left_features.shape[2],
            self.depth_channel,
            left_features.shape[-1] * 2,
        ]), trainable=False)

        for i in range(self.depth_channel):
            if i > 0:
                cost[:, :, i:, i, :left_features.shape[-1]].assign(left_features[:, :, i:, :])
                cost[:, :, i:, i, left_features.shape[-1]:].assign(right_features[:, :, :-i, :])
            else:
                cost[:, :, :, i, :left_features.shape[-1]].assign(left_features)
                cost[:, :, :, i, left_features.shape[-1]:].assign(right_features)
        
        cost = self.conv3d(cost)
        cost = tf.reshape(cost, (batch_size, w, h, -1))
        return cost


class PSMCosineLayer(layers.Layer):
    def __init__(self, max_disp=192, downsample_scale=4):
        super(PSMCosineLayer, self).__init__()
        self.max_disp = max_disp
        self.downsample_scale = downsample_scale
        self.depth_channel = int(self.max_disp / self.downsample_scale)

    def call(self, left_features, right_features):
        cost = tf.Variable(tf.zeros([
            left_features.shape[0],
            left_features.shape[1],
            left_features.shape[2],
            self.depth_channel,
        ]), trainable=False)

        for i in range(self.depth_channel):
            if i > 0:
                cost[:, :, i:, i].assign(tf.reduce_mean(left_features[:, :, i:, :] * right_features[:, :, :-i, :], axis=-1))
            else:
                cost[:, :, :, i].assign(tf.reduce_mean(left_features * right_features, axis=-1))
        return cost


class DoublePSMCosineLayer(PSMCosineLayer):
    def __init__(self, max_disp=192, downsample_scale=4):
        super(DoublePSMCosineLayer, self).__init__(max_disp=max_disp, downsample_scale=downsample_scale)
        self.depth_channel = self.depth_channel
    
    def call(self, left_features, right_features):
        b, h, w, c = left_features.shape
        # TODO:
        return 0