import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def identity(name=None):
    return layers.Lambda(lambda x: tf.identity(x, name=name))


def conv_bn_relu(filters,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 kernel_initializer=None,
                 bias_initializer=None,
                 name=None):
    """Convolution + Batch Normalization + ReLU"""
    return keras.Sequential([
        layers.ZeroPadding2D(padding),
        layers.Conv2D(filters,
                      kernel_size=kernel_size,
                      strides=stride,
                      padding='valid',
                      dilation_rate=dilation,
                      kernel_initializer=kernel_initializer,
                      bias_initializer=bias_initializer,
                      use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
    ], name=name)


def conv(filters,
         kernel_size,
         stride=1,
         padding=0,
         dilation=1,
         groups=1,
         use_bias=False,
         kernel_initializer=None,
         bias_initializer=None,
         name=None):
    """Convolution with padding"""
    return keras.Sequential([
        layers.ZeroPadding2D(padding),
        layers.Conv2D(filters,
                      kernel_size=kernel_size,
                      strides=stride,
                      padding='valid',
                      use_bias=use_bias,
                      groups=groups,
                      kernel_initializer=kernel_initializer,
                      bias_initializer=bias_initializer,
                      dilation_rate=dilation)
    ], name=name)


def conv3x3(filters,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            use_bias=False,
            kernel_initializer=None,
            bias_initializer=None,
            name=None):
    """3x3 convolution with padding"""
    return conv(filters,
                kernel_size=3,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name=name)


def conv1x1(filters,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            use_bias=False,
            kernel_initializer=None,
            bias_initializer=None,
            name=None):
    """1x1 convolution with padding"""
    return conv(filters,
                kernel_size=1,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name=name)


def conv7x7(filters,
            stride=2,
            padding=3,
            dilation=1,
            groups=1,
            use_bias=False,
            kernel_initializer=None,
            bias_initializer=None,
            name=None):
    """7x7 convolution with padding"""
    return conv(filters,
                kernel_size=7,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name=name)


def conv3d_3x3(filters,
               stride=1,
               padding=1,
               kernel_initializer=None,
               bias_initializer=None,
               name=None):
    """3D convolution with padding."""
    return keras.Sequential([
        layers.ZeroPadding3D(padding),
        layers.Conv3D(filters,
                      kernel_size=3,
                      strides=stride,
                      kernel_initializer=kernel_initializer,
                      bias_initializer=bias_initializer,
                      padding='valid')
    ], name=name)