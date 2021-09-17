import tensorflow as tf
from easydict import EasyDict as edict


def get_optimizer(schedule: tf.keras.optimizers.schedules.LearningRateSchedule,
                  cfg: edict):
    """
    """
    if schedule is not None and 'learning_rate' in cfg.optimizer.kwargs:
        del cfg.optimizer.kwargs['learning_rate']

    if cfg.optimizer.name.lower() == 'sgd':
        return tf.keras.optimizers.SGD(schedule, **(cfg.optimizer.kwargs))
    elif cfg.optimizer.name.lower() == 'adam':
        return tf.keras.optimizers.Adam(schedule, **(cfg.optimizer.kwargs))
    else:
        raise NotImplementedError(cfg.optimizer)