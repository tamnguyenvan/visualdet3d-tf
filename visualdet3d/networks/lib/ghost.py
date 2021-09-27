import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from visualdet3d.networks.lib.common import conv, identity


class GhostLayer(layers.Layer):
    def __init__(self, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostLayer, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = keras.Sequential([
            layers.AveragePooling2D(stride) if stride > 1 else identity(),
            # layers.Conv2D(init_channels, kernel_size, 1, kernel_size // 2, use_bias=False),
            conv(init_channels, kernel_size, 1, kernel_size // 2, use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU() if relu else identity(),
        ])

        self.cheap_operation = keras.Sequential([
            # layers.Conv2D(new_channels, dw_size, dw_size // 2, groups=init_channels, use_bias=True),
            conv(new_channels, dw_size, 1, dw_size // 2, groups=init_channels, use_bias=True),
            layers.BatchNormalization(),
            layers.ReLU() if relu else identity(),
        ])
    
    def call(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = tf.concat([x1, x2], axis=-1)
        return out[:, :, :, :self.oup]


class ResGhostLayer(GhostLayer):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, relu=True, stride=1):
        assert ratio > 2
        super(ResGhostLayer, self).__init__(oup-inp, kernel_size, ratio-1, dw_size,
                                            relu=relu, stride=stride)
        self.oup = oup
        if stride > 1:
            self.downsample = layers.AveragePooling2D(stride, stride)
        else:
            self.downsample = None
    
    def call(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)

        if not self.downsample is None:
            x = self.downsample(x)
        out = tf.concat([x, x1, x2], axis=-1)
        return out[:, :, :, :self.oup]