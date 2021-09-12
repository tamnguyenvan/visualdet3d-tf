import tensorflow as tf
from easydict import EasyDict as edict


def get_optimizer(cfg: edict):
    """
    """
    if cfg.optimizer.name.lower() == 'sgd':
        return tf.keras.optimizers.SGD(**(cfg.optimizer.kwargs))
    elif cfg.optimizer.name.lower() == 'adam':
        return tf.keras.optimizers.Adam(**(cfg.optimizer.kwargs))
    else:
        raise NotImplementedError(cfg.optimizer)