import tensorflow as tf


def isNaN(x):
    return x != x


class Disp2Prob(object):
    """
    Convert disparity map to matching probability volume
        Args:
            max_disp, (int): the maximum of disparity
            gt_disp, (tf.Tensor): in (..., Height, Width) layout
            start_disp (int): the start searching disparity index, usually be 0
            dilation (int): the step between near disparity index
        Outputs:
            probability, (tf.Tensor): in [BatchSize, max_disp, Height, Width] layout
    """
    def __init__(self, max_disp:int, gt_disp: tf.Tensor, start_disp:int=0, dilation:int=1):

        if not isinstance(max_disp, int):
            raise TypeError('int is expected, got {}'.format(type(max_disp)))

        if not isinstance(gt_disp, tf.Tensor):
            raise TypeError('tensor is expected, got {}'.format(type(gt_disp)))

        if not isinstance(start_disp, int):
            raise TypeError('int is expected, got {}'.format(type(start_disp)))

        if not isinstance(dilation, int):
            raise TypeError('int is expected, got {}'.format(type(dilation)))

        if tf.rank(gt_disp) == 2:  # single image H x W
            gt_disp = tf.reshape(
                gt_disp,
                (1, 1, gt_disp.shape[0], gt_disp.shape[1])
            )

        if tf.rank(gt_disp) == 3:  # multi image B x H x W
            gt_disp = tf.reshape(
                gt_disp,
                (gt_disp.shape[0], 1, gt_disp.shape[1], gt_disp.shape[2])
            )

        if tf.rank(gt_disp) == 4:
            if gt_disp.shape[1] == 1:  # mult image B x 1 x H x W
                gt_disp = gt_disp
            else:
                raise ValueError('2nd dimension size should be 1, got {}'.format(gt_disp.shape[1]))

        self.gt_disp = gt_disp
        self.max_disp = max_disp
        self.start_disp = start_disp
        self.dilation = dilation
        self.end_disp = start_disp + max_disp - 1
        self.disp_sample_number = (max_disp + dilation -1) // dilation
        self.eps = 1e-40

    def get_prob(self):
        # [BatchSize, 1, Height, Width]
        b, c, h, w = self.gt_disp.shape
        assert c == 1

        # if start_disp = 0, dilation = 1, then generate disparity candidates as [0, 1, 2, ... , max_disp-1]
        index = tf.linspace(self.start_disp, self.end_disp, self.disp_sample_number)
        index = tf.cast(index, dtype=tf.float32)
        # index = index.to(self.gt_disp.device)

        # [BatchSize, max_disp, Height, Width]
        self.index = tf.transpose(
            tf.tile(index[None, None, None, :], (b, h, w, 1)),
            (0, 3, 1, 2)
        )
        # self.index = index.repeat(b, h, w, 1).permute(0, 3, 1, 2).contiguous()

        # the gt_disp must be (start_disp, end_disp), otherwise, we have to mask it out
        mask = (self.gt_disp > self.start_disp) & (self.gt_disp < self.end_disp)
        # mask = mask.detach().type_as(self.gt_disp)
        mask = tf.cast(tf.stop_gradient(mask), dtype=self.gt_disp.dtype)
        self.gt_disp = self.gt_disp * mask

        probability = self.cal_prob()

        # let the outliers' probability to be 0
        # in case divide or log 0, we plus a tiny constant value
        probability = probability * mask + self.eps

        # in case probability is NaN
        if isNaN(tf.reduce_min(probability)) or isNaN(tf.reduce_max(probability)):
            print('Probability ==> min: {}, max: {}'.format(tf.reduce_min(probability), tf.reduce_max(probability)))
            print('Disparity Ground Truth after mask out ==> min: {}, max: {}'.format(tf.reduce_min(self.gt_disp),
                                                                                      tf.reduce_max(self.gt_disp)))
            raise ValueError(" \'probability contains NaN!")

        return probability

    def kick_invalid_half(self):
        distance = self.gt_disp - self.index
        invalid_index = distance < 0
        # after softmax, the valid index with value 1e6 will approximately get 0
        distance[invalid_index] = 1e6
        return distance

    def cal_prob(self):
        raise NotImplementedError


class LaplaceDisp2Prob(Disp2Prob):
    # variance is the diversity of the Laplace distribution
    def __init__(self, max_disp, gt_disp, variance=1, start_disp=0, dilation=1):
        super(LaplaceDisp2Prob, self).__init__(max_disp, gt_disp, start_disp, dilation)
        self.variance = variance

    def cal_prob(self):
        # 1/N * exp( - (d - d{gt}) / var), N is normalization factor, [BatchSize, max_disp, Height, Width]
        scaled_distance = ((-tf.abs(self.index - self.gt_disp)) / self.variance)
        probability = tf.math.softmax(scaled_distance, axis=1)

        return probability


class GaussianDisp2Prob(Disp2Prob):
    # variance is the variance of the Gaussian distribution
    def __init__(self, max_disp, gt_disp, variance=1, start_disp=0, dilation=1):
        super(GaussianDisp2Prob, self).__init__(max_disp, gt_disp, start_disp, dilation)
        self.variance = variance

    def calProb(self):
        # 1/N * exp( - (d - d{gt})^2 / b), N is normalization factor, [BatchSize, max_disp, Height, Width]
        distance = (tf.abs(self.index - self.gt_disp))
        scaled_distance = (- distance.pow(2.0) / self.variance)
        probability = tf.math.softmax(scaled_distance, axis=1)

        return probability


class OneHotDisp2Prob(Disp2Prob):
    # variance is the variance of the OneHot distribution
    def __init__(self, max_disp, gt_disp, variance=1, start_disp=0, dilation=1):
        super(OneHotDisp2Prob, self).__init__(max_disp, gt_disp, start_disp, dilation)
        self.variance = variance

    def getProb(self):

        # |d - d{gt}| < variance, [BatchSize, max_disp, Height, Width]
        probability = tf.less(tf.abs(self.index - self.gt_disp), self.variance).type_as(self.gt_disp)

        return probability