import tensorflow as tf
from .disp2prob import LaplaceDisp2Prob, GaussianDisp2Prob, OneHotDisp2Prob


def adaptive_pool2d(
    inputs,
    output_size,
    reduce_function=tf.reduce_max,
    data_format='channels_first'):
    """
    """
    h_bins = output_size[0]
    w_bins = output_size[1]
    if data_format == "channels_last":
        split_cols = tf.split(inputs, h_bins, axis=1)
        split_cols = tf.stack(split_cols, axis=1)
        split_rows = tf.split(split_cols, w_bins, axis=3)
        split_rows = tf.stack(split_rows, axis=3)
        out_vect = reduce_function(split_rows, axis=[2, 4])
    else:
        split_cols = tf.split(inputs, h_bins, axis=2)
        split_cols = tf.stack(split_cols, axis=2)
        split_rows = tf.split(split_cols, w_bins, axis=4)
        split_rows = tf.stack(split_rows, axis=4)
        out_vect = reduce_function(split_rows, axis=[3, 5])
    return out_vect


def adaptive_max_pool2d(
    inputs,
    output_size,
    data_format='channels_first'):
    """
    """
    return adaptive_pool2d(inputs, output_size, tf.reduce_max, data_format)


def adaptive_avg_pool2d(
    inputs,
    output_size,
    data_format='channels_first'):
    """
    """
    return adaptive_pool2d(inputs, output_size, tf.reduce_mean, data_format)


class StereoFocalLoss(object):
    """
    Under the same start disparity and maximum disparity, calculating all estimated cost volumes' loss
        Args:
            max_disp, (int): the max of Disparity. default: 192
            start_disp, (int): the start searching disparity index, usually be 0
            dilation (int): the step between near disparity index, it mainly used in gt probability volume generation
            weights, (list of float or None): weight for each scale of est_cost.
            focal_coefficient, (float): stereo focal loss coefficient, details please refer to paper. default: 0.0
            sparse, (bool): whether the ground-truth disparity is sparse, for example, KITTI is sparse, but SceneFlow is not. default: False
        Inputs:
            est_cost, (Tensor or list of Tensor): the estimated cost volume, in (BatchSize, max_disp, Height, Width) layout
            gt_disp, (Tensor): the ground truth disparity map, in (BatchSize, 1, Height, Width) layout.
            variance, (Tensor or list of Tensor): the variance of distribution, details please refer to paper, in (BatchSize, 1, Height, Width) layout.
        Outputs:
            loss, (dict), the loss of each level
        ..Note:
            Before calculate loss, the est_cost shouldn't be normalized,
              because we will use softmax for normalization
    """

    def __init__(self, max_disp=192, start_disp=0, dilation=1, weights=None, focal_coefficient=0.0, sparse=False):
        self.max_disp = max_disp
        self.start_disp = start_disp
        self.dilation = dilation
        self.weights = weights
        self.focal_coefficient = focal_coefficient
        self.sparse = sparse
        if sparse:
            # sparse disparity ==> max_pooling
            # TODO: adaptive max pooling 2d?
            self.scale_func = adaptive_max_pool2d
        else:
            # dense disparity ==> avg_pooling
            # TODO: adaptive average pooling 2d
            self.scale_func = adaptive_avg_pool2d

    def loss_per_level(self, est_cost, gt_disp, variance, dilation):
        N, C, H, W = est_cost.shape
        scaled_gt_disp = tf.identity(gt_disp)
        scale = 1.0
        if gt_disp.shape[-2] != H or gt_disp.shape[-1] != W:
            # compute scale per level and scale gt_disp
            scale = gt_disp.shape[-1] / (W * 1.0)
            scaled_gt_disp = tf.identity(gt_disp) / scale

            scaled_gt_disp = self.scale_func(scaled_gt_disp, (H, W))

        # mask for valid disparity
        # (start_disp, max disparity / scale)
        # Attention: the invalid disparity of KITTI is set as 0, be sure to mask it out
        lower_bound = self.start_disp
        upper_bound = lower_bound + int(self.max_disp/scale)
        mask = (scaled_gt_disp > lower_bound) & (scaled_gt_disp < upper_bound)
        mask = tf.cast(tf.stop_gradient(mask), dtype=scaled_gt_disp.dtype)
        if tf.reduce_sum(mask) < 1.0:
            print('Stereo focal loss: there is no point\'s '
                  'disparity is in [{},{})!'.format(lower_bound, upper_bound))
            scaled_gt_prob = tf.zeros_like(est_cost)  # let this sample have loss with 0
        else:
            # transfer disparity map to probability map
            mask_scaled_gt_disp = scaled_gt_disp * mask
            scaled_gt_prob = LaplaceDisp2Prob(
                int(self.max_disp/scale),
                mask_scaled_gt_disp,
                variance=variance,
                start_disp=self.start_disp,
                dilation=dilation
            ).get_prob()

        # stereo focal loss
        est_prob = tf.math.log_softmax(est_cost, axis=1)
        weight = tf.cast(
            tf.pow(1.0 - scaled_gt_prob, -self.focal_coefficient),
            dtype=scaled_gt_prob.dtype
        )
        loss = tf.reduce_mean(
            tf.reduce_sum(
                -((scaled_gt_prob * est_prob) * weight * tf.cast(mask, tf.float32)),
                axis=1,
                keepdims=True
            )
        )

        return loss

    def __call__(self, est_cost, gt_disp, variance):
        if not isinstance(est_cost, (list, tuple)):
            est_cost = [est_cost]

        if self.weights is None:
            self.weights = 1.0

        if not isinstance(self.weights, (list, tuple)):
            self.weights = [self.weights] * len(est_cost)

        if not isinstance(self.dilation, (list, tuple)):
            self.dilation = [self.dilation] * len(est_cost)

        if not isinstance(variance, (list, tuple)):
            variance = [variance] * len(est_cost)

        # compute loss for per level
        loss_all_level = []
        for est_cost_per_lvl, var, dt in zip(est_cost, variance, self.dilation):
            loss_all_level.append(
                self.loss_per_level(est_cost_per_lvl, gt_disp, var, dt))

        # re-weight loss per level
        loss = 0
        for i, loss_per_level in enumerate(loss_all_level):
            loss += self.weights[i] * loss_per_level

        return loss

    def __repr__(self):
        repr_str = '{}\n'.format(self.__class__.__name__)
        repr_str += ' ' * 4 + 'Max Disparity: {}\n'.format(self.max_disp)
        repr_str += ' ' * 4 + 'Start disparity: {}\n'.format(self.start_disp)
        repr_str += ' ' * 4 + 'Dilation rate: {}\n'.format(self.dilation)
        repr_str += ' ' * 4 + 'Loss weight: {}\n'.format(self.weights)
        repr_str += ' ' * 4 + 'Focal coefficient: {}\n'.format(self.focal_coefficient)
        repr_str += ' ' * 4 + 'Disparity is sparse: {}\n'.format(self.sparse)

        return repr_str

    @property
    def name(self):
        return 'StereoFocalLoss'