from tensorflow import keras
from tensorflow.keras import layers
from tensorflow_addons.layers.deformable_conv2d import DeformableConv2D


def deform_conv2d(filters,
                  kernel_size=3,
                  strides=1,
                  padding=1):
    """Deformable convolution 2D with padding"""
    return keras.Sequential([
        layers.ZeroPadding2D(padding),
        DeformableConv2D(filters,
                         kernel_size=kernel_size,
                         strides=strides,
                         padding='valid')
    ])