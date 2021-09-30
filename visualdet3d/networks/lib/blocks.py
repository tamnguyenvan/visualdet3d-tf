import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class AnchorFlatten(layers.Layer):
    """
        Module for anchor-based network outputs,
        Init args:
            num_output: number of output channel for each anchor.
        Forward args:
            x: torch.tensor of shape [B, H, W, num_anchors * output_channel]
        Forward return:
            x : torch.tensor of shape [B, num_anchors * H * W, output_channel]
    """
    def __init__(self, num_output_channel):
        super(AnchorFlatten, self).__init__()
        self.num_output_channel = num_output_channel

    def call(self, x):
        # x = tf.transpose(0, 2, 3, 1)
        x = tf.reshape(x, (x.shape[0], -1, self.num_output_channel))
        # x = x.contiguous().view(x.shape[0], -1, self.num_output_channel)
        return x