from typing import Optional

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from visualdet3d.models.lib.disparity_loss import stereo_focal_loss


class SigmoidFocalLoss(losses.Loss):
    def __init__(self, gamma=0.0, balance_weights=tf.constant([1.0], dtype=tf.float32)):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.balance_weights = balance_weights

    def call(self,
             classification: tf.Tensor, 
             targets: tf.Tensor, 
             gamma: Optional[float]=None, 
             balance_weights: Optional[tf.Tensor]=None):
        """
        Args
          classification  :[..., num_classes]  linear output
          targets         :[..., num_classes] == -1(ignored), 0, 1

        Returns
          cls_loss        :[..., num_classes]  loss with 0 in trimmed or ignored indexes 
        """
        if gamma is None:
            gamma = self.gamma
        if balance_weights is None:
            balance_weights = self.balance_weights

        probs = tf.sigmoid(classification)  #[B, N, 1]
        focal_weight = tf.where(tf.equal(targets, 1.), 1. - probs, probs)
        focal_weight = tf.pow(focal_weight, gamma)

        bce = -(targets * tf.math.log_sigmoid(classification)) * balance_weights \
            - ((1-targets) * tf.math.log_sigmoid(-classification)) #[B, N, 1]
        cls_loss = focal_weight * bce

        ## neglect  0.3 < iou < 0.4 anchors
        cls_loss = tf.where(tf.not_equal(targets, -1.0), cls_loss, tf.zeros(cls_loss.shape))

        ## clamp over confidence and correct ones to prevent overfitting
        cls_loss = tf.where(tf.less(cls_loss, 1e-5), tf.zeros(cls_loss.shape), cls_loss)  #0.02**2 * log(0.98) = 8e-6

        return cls_loss


class SoftmaxFocalLoss(losses.Loss):
    def __init__(self):
        super(SoftmaxFocalLoss, self).__init__()

    def call(self,
             classification: tf.Tensor, 
             targets: tf.Tensor, 
             gamma: float, 
             balance_weights: tf.Tensor):
        ## Calculate focal loss weights
        probs = tf.math.softmax(classification, axis=-1)
        focal_weight = tf.where(tf.equal(targets, 1.), 1. - probs, probs)
        focal_weight = tf.pow(focal_weight, gamma)

        bce = -(targets * tf.math.log_softmax(classification, dim=1))

        cls_loss = focal_weight * bce

        ## neglect  0.3 < iou < 0.4 anchors
        cls_loss = tf.where(tf.not_equal(targets, -1.0), cls_loss, tf.zeros(cls_loss.shape))

        ## clamp over confidence and correct ones to prevent overfitting
        cls_loss = tf.where(tf.less(cls_loss, 1e-5), tf.zeros(cls_loss.shape), cls_loss) #0.02**2 * log(0.98) = 8e-6
        cls_loss = cls_loss * balance_weights
        return cls_loss


class ModifiedSmoothL1Loss(losses.Loss):
    def __init__(self, l1_regression_alpha: float):
        super(ModifiedSmoothL1Loss, self).__init__()
        self.alpha = l1_regression_alpha

    def call(self, normed_targets: tf.Tensor, pos_reg: tf.Tensor):
        regression_diff = tf.abs(normed_targets - pos_reg)  #[K, 12]

        ## Smoothed-L1 formula:
        regression_loss = tf.where(
            tf.less_equal(regression_diff, 1.0 / self.alpha),
            0.5 * self.alpha * tf.pow(regression_diff, 2),
            regression_diff - 0.5 / self.alpha
        )
        ## clipped to avoid overfitting
        regression_loss = tf.where(
           tf.less_equal(regression_diff, 0.01),
           tf.zeros_like(regression_loss),
           regression_loss
        )

        return regression_loss


class IoULoss(losses.Loss):
    def __init__(self):
        super(IoULoss, self).__init__()

    def call(self, preds: tf.Tensor, targets: tf.Tensor, eps:float=1e-8):
        """IoU Loss

        Args:
            preds (torch.Tensor): [x1, y1, x2, y2] predictions [*, 4]
            targets (torch.Tensor): [x1, y1, x2, y2] targets [*, 4]

        Returns:
            torch.Tensor: [-log(iou)] [*]
        """
        
        # overlap
        lt = tf.reduce_max(preds[..., :2], targets[..., :2])
        rb = tf.reduce_min(preds[..., 2:], targets[..., 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1]

        # union
        ap = (preds[..., 2] - preds[..., 0]) * (preds[..., 3] - preds[..., 1])
        ag = (targets[..., 2] - targets[..., 0]) * (targets[..., 3] - targets[..., 1])
        union = ap + ag - overlap + eps

        # IoU
        ious = overlap / union
        ious = tf.clip_by_value(ious, clip_value_min=eps)
        return -ious.log()


class DisparityLoss(losses.Loss):
    def __init__(self, maxdisp: int=64):
        super(DisparityLoss, self).__init__()
        self.criterion = stereo_focal_loss.StereoFocalLoss(maxdisp)

    def call(self, x: tf.Tensor, label: tf.Tensor):
        label = tf.expand_dims(label, axis=1)
        loss = self.criterion(x, label, variance=0.5)
        return loss