from typing import Union
from easydict import EasyDict as edict

import numpy as np
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.optimizers.schedules import (
    ExponentialDecay, CosineDecay
)


class PolyLR(LearningRateSchedule):
    """
    """
    def __init__(self, learning_rate=1e-3, gamma=0.9, n_iteration=-1, verbose=False):
        super(PolyLR, self).__init__()
        self.learning_rate = learning_rate
        self.step_size = n_iteration
        self.gamma = gamma
        self.verbose = verbose

    def __call__(self, step):
        decay = (1 - step / float(self.step_size)) ** self.gamma
        lr = self.learning_rate * decay
        if self.verbose:
            print('Learning rate:', lr.numpy())
        return lr


class StepLR(LearningRateSchedule):
    """
    """
    def __init__(self, learning_rate, step_size, gamma=0.1, verbose=False):
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma
        self.verbose = verbose
    
    def __call__(self, step):
        lr = self.learning_rate * self.gamma ** (step // self.step_size)
        if self.verbose:
            print('Learning rate:', lr.numpy())
        return lr


class MultiStepLR(LearningRateSchedule):
    """
    """
    def __init__(self, learning_rate, milestones, gamma, verbose=False):
        self.learning_rate = learning_rate 
        self.milestones = milestones
        self.gamma = gamma
        self.verbose = verbose
    
    def __call__(self, step):
        lr = self.learning_rate * self.gamma ** int(np.digitize(step, self.milestones))
        if self.verbose:
            print('Learning rate:', lr.numpy())
        return lr


class ExponentialLR(LearningRateSchedule):
    def __init__(self, learning_rate, decay_steps, gamma):
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.gamma = gamma
        self.scheduler = ExponentialDecay(learning_rate, decay_steps, gamma)
    
    def __call__(self, step):
        lr = self.scheduler(step)
        if self.verbose:
            print('Learning rate:', lr.numpy())
        return lr


class CosineAnnealingLR(LearningRateSchedule):
    """
    """
    def __init__(self, learning_rate, decay_steps, verbose=False):
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.verbose = verbose
        self.scheduler = CosineDecay(learning_rate, decay_steps)
    
    def __call__(self, step):
        lr = self.scheduler(step)
        print(lr)
        if self.verbose:
            print('Learning rate:', lr.numpy())
        return lr


def get_scheduler(cfg: Union[edict, None]):
    """
    """
    kwargs = cfg.scheduler.kwargs
    kwargs.update({'learning_rate': cfg.optimizer.kwargs.lr})
    if cfg is None:
        return ExponentialLR(**kwargs)
    if cfg.scheduler.type_name.lower() == 'StepLR'.lower():
        return StepLR(**kwargs)
    if cfg.scheduler.type_name.lower() == 'MultiStepLR'.lower():
        return MultiStepLR(**kwargs)
    if cfg.scheduler.type_name.lower() == 'ExponentialLR'.lower():
        return ExponentialLR(**kwargs)
    if cfg.scheduler.type_name.lower() == 'CosineAnnealingLR'.lower():
        return CosineAnnealingLR(**kwargs)
    if cfg.scheduler.type_name.lower() == 'PolyLR'.lower():
        return PolyLR(**kwargs)
    
    raise NotImplementedError(cfg)