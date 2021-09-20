from typing import Union
from easydict import EasyDict as edict

import numpy as np
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.optimizers.schedules import (
    ExponentialDecay, PiecewiseConstantDecay, PolynomialDecay
)


class PolyLR(LearningRateSchedule):
    """
    """
    def __init__(self, lr=1e-3, gamma=0.9, n_iteration=-1, verbose=False):
        super(PolyLR, self).__init__()
        self.lr = lr
        self.step_size = n_iteration
        self.gamma = gamma
        self.verbose = verbose

    def __call__(self, step):
        decay = (1 - step / float(self.step_size)) ** self.gamma
        lr = self.lr * decay
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


class GradualWarmupScheduler(LearningRateSchedule):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    
    From:
        https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/master/warmup_scheduler/scheduler.py
    """

    def __init__(self, optimizer, multiplier:float, total_epoch:int, after_cfg=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = get_scheduler(after_cfg, optimizer)
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != optim.lr_scheduler.ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


def get_scheduler(cfg: Union[edict, None]):
    """
    """
    if cfg is None:
        return ExponentialDecay(cfg.optimizer.kwargs.lr)
    if cfg.scheduler.name.lower() == 'StepLR'.lower():
        return StepLR(**(cfg.scheduler.kwargs))
    if cfg.scheduler.name.lower() == 'MultiStepLR'.lower():
        return MultiStepLR(**cfg.scheduler.kwargs)
    if cfg.scheduler.name.lower() == 'ExponentialLR'.lower():
        # return ExponentialLR(optimizer, **cfg.keywords)
        return ExponentialDecay(cfg.optimizer.kwargs.lr)
    if cfg.scheduler.name.lower() == 'CosineAnnealingLR'.lower():
        # return CosineAnnealingLR(optimizer, **cfg.keywords)
        return
    if cfg.scheduler.name.lower() == 'PolyLR'.lower():
        # return PolyLR(optimizer, **cfg.keywords)
        return 
    if cfg.scheduler.name.lower() == 'GradualWarmupScheduler'.lower():
        # return GradualWarmupScheduler(optimizer, **cfg.keywords)
        return
    
    raise NotImplementedError(cfg)