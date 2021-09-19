"""Backbone networks implementation"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from visualdet3d.models.layers.common import conv1x1, conv3x3, conv7x7


class BasicBlock(layers.Layer):
    """Pre-activation version of the Basic Block."""
    expansion = 1

    def __init__(self, filters, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(filters, stride)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = conv3x3(filters, dilation=dilation)
        self.bn2 = layers.BatchNormalization()

        self.downsample = downsample
        self.stride = stride
    
    def call(self, x):
        shortcut = x
        out = tf.nn.relu(self.bn1(x))
        if self.downsample is not None:
            shortcut = self.downsample(out)
        
        out = self.conv1(out)
        out = self.conv2(tf.nn.relu(self.bn2(out)))
        out += shortcut
        return out


class Bottleneck(layers.Layer):
    """Pre-activation version of the original Bottleneck module."""
    expansion = 4
    
    def __init__(self, filters, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(filters)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = conv3x3(filters, stride=stride, dilation=dilation)
        self.bn2 = layers.BatchNormalization()
        self.conv3 = conv1x1(filters*4)
        self.bn3 = layers.BatchNormalization()
        self.downsample = downsample
        self.stride = stride
    
    def call(self, x):
        shortcut = x
        out = tf.nn.relu(self.bn1(x))
        if self.downsample is not None:
            shortcut = self.downsample(out)
        
        out = self.conv1(out)
        out = self.conv2(tf.nn.relu(self.bn2(out)))
        out = self.conv3(tf.nn.relu(self.bn3(out)))
        out += shortcut
        return out


class ResNetBase(keras.Model):
    def __init__(self,
                 block,
                 num_blocks,
                 pretrained=True,
                 frozen_stages=-1,
                 num_stages=3,
                 norm_eval=True,
                 dilations=[1, 1, 1],
                 num_filters=[64, 128, 256, 512],
                 strides=[1, 2, 2, 2],
                 out_indices=[-1, 0, 1, 2, 3],
                 name=None):
        super(ResNetBase, self).__init__(name=name)
        self.pretrained = pretrained
        self.frozen_stages = frozen_stages
        self.num_stages = num_stages
        self.norm_eval = norm_eval
        self.dilations = dilations
        self.out_indices = out_indices
        self.filters = 64

        self.conv1 = conv7x7(64, stride=2, padding=3, name='init_conv')
        self.bn1 = layers.BatchNormalization(name='init_bn')
        self.pool = layers.MaxPooling2D(3, (2, 2), padding='same', name='init_pool')

        for i in range(4):
            stride = strides[i]
            setattr(self, f'layer{i+1}',
                    self._make_layer(block, num_filters[i], num_blocks[i], stride))

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, name='layer_1')
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, name='layer_2')
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, name='layer_3')
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, name='layer_4')

    def _make_layer(self, block, filters, num_blocks, stride, name=None):
        strides = [stride] + [1] * (num_blocks - 1)
        layer_list = []
        for stride in strides:
            downsample = None
            if stride != 1 or self.filters != filters * block.expansion:
                downsample = keras.Sequential([
                    conv1x1(filters * block.expansion, stride=stride),
                    layers.BatchNormalization(),
                ])
            layer_list.append(block(filters, stride, downsample))
            self.filters = filters * block.expansion
        
        return keras.Sequential(layer_list, name=name)
    
    def call(self, x):
        outs = []
        out = self.conv1(x)
        out = tf.nn.relu(self.bn1(out))
        if -1 in self.out_indices:
            outs.append(out)
        out = self.pool(out)
        for i in range(4):
            layer = getattr(self, f'layer{i+1}')
            out = layer(out)
            if i in self.out_indices:
                outs.append(out)
        return outs


def ResNet(depth,
           input_shape=(288, 1280, 3),
           pretrained=True,
           frozen_stages=-1,
           num_stages=3,
           out_indices=[0, 1, 2],
           norm_eval=True,
           dilations=[1, 1, 1],
           name=None):
    """
    """
    kwargs = {
        'pretrained': pretrained,
        'frozen_stages': frozen_stages,
        'num_stages': num_stages,
        'out_indices': out_indices,
        'norm_eval': norm_eval,
        'dilations': dilations,
    }
    if depth == 18:
        model = ResNetBase(BasicBlock, [2, 2, 2, 2], name=name, **kwargs)
        model.build([1, *input_shape])
    elif depth == 34:
        model = ResNetBase(BasicBlock, [3, 4, 6, 3], name=name, **kwargs)
        model.build([1, *input_shape])
    elif depth == 50:
        model = ResNetBase(Bottleneck, [3, 4, 23, 3], name=name, **kwargs)
        model.build([1, *input_shape])
    elif depth == 152:
        model = ResNetBase(Bottleneck, [3, 8, 36, 3], name=name, **kwargs)
        model.build([1, *input_shape])
    else:
        raise NotImplementedError(f'Invalid depth: {depth}')
    return model
        

if __name__ == '__main__':
    input_shape = (224, 224, 3)
    model = ResNet(depth=34)
    model.summary()

    dummy_inputs = tf.random.normal((1, 224, 224, 3))
    outs = model(dummy_inputs)
    for out in outs:
        print(out.shape)