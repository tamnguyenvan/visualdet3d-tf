import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from easydict import EasyDict as edict
from tensorflow.python.keras import initializers

from visualdet3d.networks.heads.losses import SigmoidFocalLoss, ModifiedSmoothL1Loss
from visualdet3d.networks.heads.anchors import Anchors
from visualdet3d.networks.backbones.resnet import BasicBlock
from visualdet3d.networks.utils import calc_iou, BackProjection, ClipBoxes
from visualdet3d.networks.lib.common import conv3x3, conv_bn_relu
from visualdet3d.networks.lib.blocks import AnchorFlatten
from visualdet3d.networks.lib.fast_utils.hill_climbing import post_opt
# from visualdet3d.networks.lib.ops.deform_conv import deform_conv2d


class AnchorBasedDetection3DHead(layers.Layer):
    def __init__(self,
                #  num_features_in: int=1024,
                 num_classes: int=3,
                 num_regression_loss_terms=12,
                 preprocessed_path: str='',
                 anchors_cfg: edict=edict(),
                 layer_cfg: edict=edict(),
                 loss_cfg: edict=edict(),
                 test_cfg: edict=edict(),
                 read_precompute_anchor:bool=True,
                 name=None):
        super(AnchorBasedDetection3DHead, self).__init__(name=name)
        self.anchors = Anchors(preprocessed_path=preprocessed_path,
                               read_config_file=read_precompute_anchor,
                               **anchors_cfg)
        
        self.num_classes = num_classes
        self.num_regression_loss_terms=num_regression_loss_terms
        self.decode_before_loss = getattr(loss_cfg, 'decode_before_loss', False)
        self.loss_cfg = loss_cfg
        self.test_cfg  = test_cfg
        self.build_loss(**loss_cfg)
        self.backprojector = BackProjection()
        self.clipper = ClipBoxes()

        if getattr(layer_cfg, 'num_anchors', None) is None:
            layer_cfg['num_anchors'] = self.anchors.num_anchors
        self.init_layers(**layer_cfg)

    def init_layers(self,
                    # num_features_in,
                    num_anchors: int,
                    num_cls_output: int,
                    num_reg_output: int,
                    cls_feature_size: int=1024,
                    reg_feature_size: int=1024,
                    **kwargs):

        self.cls_feature_extraction = keras.Sequential([
            conv3x3(cls_feature_size, padding=1),
            layers.Dropout(0.3),
            layers.ReLU(),
            conv3x3(cls_feature_size, padding=1),
            layers.Dropout(0.3),
            layers.ReLU(),
            
            conv3x3(num_anchors*num_cls_output,
                    padding=1,
                    kernel_initializer=keras.initializers.Zeros(),
                    bias_initializer=keras.initializers.Zeros()),
            AnchorFlatten(num_cls_output),

        ])
        # self.cls_feature_extraction[-2].weight.data.fill_(0)
        # self.cls_feature_extraction[-2].bias.data.fill_(0)

        self.reg_feature_extraction = keras.Sequential([
            # TODO: ModulatedDeformConvPack
            # deform_conv2d(reg_feature_size, 3, padding=1),
            conv3x3(reg_feature_size, padding=1),
            layers.BatchNormalization(),
            layers.ReLU(),
            conv3x3(reg_feature_size, padding=1),
            layers.BatchNormalization(),
            layers.ReLU(),
            
            conv3x3(num_anchors*num_reg_output,
                    padding=1,
                    kernel_initializer=keras.initializers.Zeros(),
                    bias_initializer=keras.initializers.Zeros()),
            AnchorFlatten(num_reg_output),
        ])

        # TODO:
        # self.reg_feature_extraction[-2].weight.data.fill_(0)
        # self.reg_feature_extraction[-2].bias.data.fill_(0)

    def call(self, inputs):
        with tf.name_scope('cls_feature_extraction'):
            cls_preds = self.cls_feature_extraction(inputs['features'])
        
        with tf.name_scope('reg_feature_extraction'):
            reg_preds = self.reg_feature_extraction(inputs['features'])

        return cls_preds, reg_preds
        
    def build_loss(self, focal_loss_gamma=0.0, balance_weight=[0], L1_regression_alpha=9, **kwargs):
        self.focal_loss_gamma = focal_loss_gamma
        self.balance_weights = tf.constant(balance_weight, dtype=tf.float32)
        self.loss_cls = SigmoidFocalLoss(
            gamma=focal_loss_gamma,
            balance_weights=self.balance_weights,
            reduction=tf.keras.losses.Reduction.NONE
        )
        self.loss_bbox = ModifiedSmoothL1Loss(
            L1_regression_alpha,
            reduction=tf.keras.losses.Reduction.NONE
        )

        regression_weight = kwargs.get("regression_weight",
                                       [1 for _ in range(self.num_regression_loss_terms)]) #default 12 only use in 3D
        self.regression_weight = tf.constant(regression_weight, dtype=tf.float32)

        self.alpha_loss = keras.losses.BinaryCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE
        )

    def _assign(self, anchor, annotation, 
                    bg_iou_threshold=0.0,
                    fg_iou_threshold=0.5,
                    min_iou_threshold=0.0,
                    match_low_quality=True,
                    gt_max_assign_all=True,
                    **kwargs):
        """
            anchor: [N, 4]
            annotation: [num_gt, 4]:
        """
        N = anchor.shape[0]
        num_gt = annotation.shape[0]
        assigned_gt_inds = tf.fill((N,), -1)
        max_overlaps = tf.zeros((N,), dtype=anchor.dtype)
        assigned_labels = tf.fill((N,), -1)

        if num_gt == 0:
            assigned_gt_inds = tf.fill((N,), 0)
            return_dict = dict(
                num_gt=num_gt,
                assigned_gt_inds=assigned_gt_inds,
                max_overlaps=max_overlaps,
                labels=assigned_labels
            )
            return return_dict

        IoU = calc_iou(anchor, annotation[:, :4]) # num_anchors x num_annotations

        # max for anchor
        max_overlaps = tf.reduce_max(IoU, axis=1)
        argmax_overlaps = tf.argmax(IoU, axis=1)

        # max for gt
        gt_max_overlaps = tf.reduce_max(IoU, axis=0)
        gt_argmax_overlaps = tf.argmax(IoU, axis=0)

        # assign negative
        indices = tf.where((max_overlaps >=0) & (max_overlaps < bg_iou_threshold))
        updates = tf.zeros(indices.shape[0], dtype=assigned_gt_inds.dtype)
        assigned_gt_inds = tf.tensor_scatter_nd_update(assigned_gt_inds, indices, updates)

        # assign positive
        pos_inds = max_overlaps >= fg_iou_threshold
        indices = tf.where(pos_inds)
        updates = tf.cast(argmax_overlaps[pos_inds] + 1, dtype=assigned_gt_inds.dtype)
        assigned_gt_inds = tf.tensor_scatter_nd_update(assigned_gt_inds, indices, updates)

        if match_low_quality:
            for i in range(num_gt):
                if gt_max_overlaps[i] >= min_iou_threshold:
                    if gt_max_assign_all:
                        max_iou_inds = IoU[:, i] == gt_max_overlaps[i]
                        indices = tf.where(max_iou_inds)
                        updates = tf.cast(tf.fill(indices.shape[0], i+1), dtype=assigned_gt_inds.dtype)
                        assigned_gt_inds = tf.tensor_scatter_nd_update(assigned_gt_inds, indices, updates)
                    else:
                        indices = tf.reshape(gt_argmax_overlaps, (-1, 1))
                        updates = tf.cast(tf.fill(indices.shape[0], i+1), dtype=assigned_gt_inds.dtype)
                        assigned_gt_inds = tf.tensor_scatter_nd_update(assigned_gt_inds, indices, updates)

        assigned_labels = tf.cast(tf.fill(N, -1), dtype=assigned_gt_inds.dtype)
        pos_inds = tf.cast(tf.squeeze(tf.where(assigned_gt_inds > 0), axis=1), dtype=tf.int32)
        if tf.size(pos_inds) > 0:
            # assigned_labels[pos_inds] = annotation[assigned_gt_inds[pos_inds] - 1, 4].long()
            indices = tf.reshape(pos_inds, (-1, 1))
            ann_inds = tf.gather(assigned_gt_inds, pos_inds) - 1
            updates = tf.cast(
                tf.gather_nd(annotation, tf.stack([ann_inds, tf.fill(ann_inds.shape[0], 4)], axis=1)),
                tf.int32
            )
            # assigned_labels[pos_inds] = tf.cast(annotation[assigned_gt_inds[pos_inds] - 1, 4], tf.int32)
            assigned_labels = tf.tensor_scatter_nd_update(assigned_labels, indices, updates)

        return_dict = dict(
            num_gt=num_gt,
            assigned_gt_inds=assigned_gt_inds,
            max_overlaps=max_overlaps,
            labels=assigned_labels
        )
        return return_dict

    def _encode(self, sampled_anchors, sampled_gt_bboxes, selected_anchors_3d):
        assert sampled_anchors.shape[0] == sampled_gt_bboxes.shape[0]

        # sampled_anchors = sampled_anchors.float()
        sampled_anchors = tf.cast(sampled_anchors, tf.float32)
        # sampled_gt_bboxes = sampled_gt_bboxes.float()
        sampled_gt_bboxes = tf.cast(sampled_gt_bboxes, tf.float32)
        px = (sampled_anchors[..., 0] + sampled_anchors[..., 2]) * 0.5
        py = (sampled_anchors[..., 1] + sampled_anchors[..., 3]) * 0.5
        pw = sampled_anchors[..., 2] - sampled_anchors[..., 0]
        ph = sampled_anchors[..., 3] - sampled_anchors[..., 1]

        gx = (sampled_gt_bboxes[..., 0] + sampled_gt_bboxes[..., 2]) * 0.5
        gy = (sampled_gt_bboxes[..., 1] + sampled_gt_bboxes[..., 3]) * 0.5
        gw = sampled_gt_bboxes[..., 2] - sampled_gt_bboxes[..., 0]
        gh = sampled_gt_bboxes[..., 3] - sampled_gt_bboxes[..., 1]

        targets_dx = (gx - px) / pw
        targets_dy = (gy - py) / ph
        # targets_dw = torch.log(gw / pw)
        # targets_dh = torch.log(gh / ph)
        targets_dw = tf.math.log(gw / pw)
        targets_dh = tf.math.log(gh / ph)

        targets_cdx = (sampled_gt_bboxes[:, 5] - px) / pw
        targets_cdy = (sampled_gt_bboxes[:, 6] - py) / ph

        targets_cdz = (sampled_gt_bboxes[:, 7] - selected_anchors_3d[:, 0, 0]) / selected_anchors_3d[:, 0, 1]
        targets_cd_sin = (tf.math.sin(sampled_gt_bboxes[:, 11] * 2) - selected_anchors_3d[:, 1, 0]) / selected_anchors_3d[:, 1, 1]
        targets_cd_cos = (tf.math.cos(sampled_gt_bboxes[:, 11] * 2) - selected_anchors_3d[:, 2, 0]) / selected_anchors_3d[:, 2, 1]
        targets_w3d = (sampled_gt_bboxes[:, 8]  - selected_anchors_3d[:, 3, 0]) / selected_anchors_3d[:, 3, 1]
        targets_h3d = (sampled_gt_bboxes[:, 9]  - selected_anchors_3d[:, 4, 0]) / selected_anchors_3d[:, 4, 1]
        targets_l3d = (sampled_gt_bboxes[:, 10] - selected_anchors_3d[:, 5, 0]) / selected_anchors_3d[:, 5, 1]

        targets = tf.stack((targets_dx, targets_dy, targets_dw, targets_dh, 
                         targets_cdx, targets_cdy, targets_cdz,
                         targets_cd_sin, targets_cd_cos,
                         targets_w3d, targets_h3d, targets_l3d), axis=1)

        # stds = targets.new([0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 1, 1, 1, 1, 1, 1])
        stds = tf.constant([0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 1, 1, 1, 1, 1, 1])

        # targets = targets.div_(stds)
        targets = tf.truediv(targets, stds)

        targets_alpha_cls = tf.cast(tf.math.cos(sampled_gt_bboxes[:, 11:12]) > 0, tf.float32)
        return targets, targets_alpha_cls  #[N, 4]

    def _decode(self, boxes, deltas, anchors_3d_mean_std, label_index, alpha_score):
        std = tf.constant([0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 1, 1, 1, 1, 1, 1], dtype=tf.float32)
        widths  = boxes[..., 2] - boxes[..., 0]
        heights = boxes[..., 3] - boxes[..., 1]
        ctr_x   = boxes[..., 0] + 0.5 * widths
        ctr_y   = boxes[..., 1] + 0.5 * heights

        dx = deltas[..., 0] * std[0]
        dy = deltas[..., 1] * std[1]
        dw = deltas[..., 2] * std[2]
        dh = deltas[..., 3] * std[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w     = tf.exp(dw) * widths
        pred_h     = tf.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        one_hot_mask = tf.cast(tf.one_hot(label_index, anchors_3d_mean_std.shape[1]), tf.bool)
        selected_mean_std = anchors_3d_mean_std[one_hot_mask] #[N]
        mask = selected_mean_std[:, 0, 0] > 0
        
        cdx = deltas[..., 4] * std[4]
        cdy = deltas[..., 5] * std[5]
        pred_cx1 = ctr_x + cdx * widths
        pred_cy1 = ctr_y + cdy * heights
        pred_z   = deltas[...,6] * selected_mean_std[:, 0, 1] + selected_mean_std[:,0, 0]  #[N, 6]
        pred_sin = deltas[...,7] * selected_mean_std[:, 1, 1] + selected_mean_std[:,1, 0] 
        pred_cos = deltas[...,8] * selected_mean_std[:, 2, 1] + selected_mean_std[:,2, 0] 
        pred_alpha = tf.math.atan2(pred_sin, pred_cos) / 2.0

        pred_w = deltas[...,9]  * selected_mean_std[:, 3, 1] + selected_mean_std[:,3, 0]
        pred_h = deltas[...,10] * selected_mean_std[:,4, 1] + selected_mean_std[:,4, 0]
        pred_l = deltas[...,11] * selected_mean_std[:,5, 1] + selected_mean_std[:,5, 0]

        pred_boxes = tf.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2,
                                    pred_cx1, pred_cy1, pred_z,
                                    pred_w, pred_h, pred_l, pred_alpha], axis=1)

        alpha_score_mask = alpha_score[:, 0] < 0.5
        alpha_score_inds = tf.cast(tf.squeeze(tf.where(alpha_score_mask), 1), tf.int32)
        inds = tf.stack([
            alpha_score_inds,
            tf.fill(alpha_score_inds.shape[0], pred_boxes.shape[-1]-1)
        ], axis=1)
        # pred_boxes[alpha_score[:, 0] < 0.5, -1] += np.pi
        updates = tf.fill(alpha_score_inds.shape[0], tf.constant(math.pi))
        pred_boxes = tf.tensor_scatter_nd_add(pred_boxes, inds, updates)

        return pred_boxes, mask
        

    def _sample(self, assignment_result, anchors, gt_bboxes):
        """
            Pseudo sampling
        """
        # pos_inds = torch.nonzero(
        #         assignment_result['assigned_gt_inds'] > 0, as_tuple=False
        #     ).unsqueeze(-1).unique()
        pos_inds = tf.unique(
            tf.squeeze(tf.where(assignment_result['assigned_gt_inds'] > 0), 1))[0]
        pos_inds = tf.cast(pos_inds, dtype=tf.int32)
        # neg_inds = torch.nonzero(
        #         assignment_result['assigned_gt_inds'] == 0, as_tuple=False
        #     ).unsqueeze(-1).unique()
        neg_inds = tf.unique(
            tf.squeeze(tf.where(assignment_result['assigned_gt_inds'] == 0), 1))[0]
        neg_inds = tf.cast(neg_inds, dtype=tf.int32)
        # gt_flags = anchors.new_zeros(anchors.shape[0], dtype=torch.uint8) #
        gt_flags = tf.zeros(anchors.shape[0], dtype=tf.uint8)

        pos_assigned_gt_inds = assignment_result['assigned_gt_inds'] - 1

        # if gt_bboxes.numel() == 0:
        if tf.size(gt_bboxes) == 0:
            # pos_gt_bboxes = gt_bboxes.new_zeros([0, 4])
            pos_gt_bboxes = tf.zeros((0, 4), dtype=gt_bboxes.dtype)
        else:
            gt_inds = tf.gather(pos_assigned_gt_inds, pos_inds)
            pos_gt_bboxes = tf.gather(
                gt_bboxes,
                gt_inds
            )
            # pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds[pos_inds], :]
        return_dict = dict(
            pos_inds=pos_inds,
            neg_inds=neg_inds,
            # pos_bboxes=anchors[pos_inds],
            pos_bboxes=tf.gather(anchors, pos_inds),
            # neg_bboxes=anchors[neg_inds],
            neg_bboxes=tf.gather(anchors, neg_inds),
            pos_gt_bboxes=pos_gt_bboxes,
            # pos_assigned_gt_inds=pos_assigned_gt_inds[pos_inds],
            pos_assigned_gt_inds=tf.gather(pos_assigned_gt_inds, pos_inds),
        )
        return return_dict

    def _post_process(self, scores, bboxes, labels, P2s):
        
        N = len(scores)
        bbox2d = bboxes[:, 0:4]
        bbox3d = bboxes[:, 4:] #[cx, cy, z, w, h, l, alpha]

        bbox3d_state_3d = self.backprojector(bbox3d, P2s[0]) #[x, y, z, w, h, l, alpha]
        for i in range(N):
            if bbox3d_state_3d[i, 2] > 3 and labels[i] == 0:
                bbox3d[i] = post_opt(
                    bbox2d[i], bbox3d_state_3d[i], P2s[0],
                    bbox3d[i, 0], bbox3d[i, 1]
                )
        bboxes = tf.concat([bbox2d, bbox3d], axis=-1)
        return scores, bboxes, labels

    def get_anchor(self, img_batch, P2, training=False):
        is_filtering = getattr(self.loss_cfg, 'filter_anchor', True)
        if not training:
            is_filtering = getattr(self.test_cfg, 'filter_anchor', is_filtering)

        anchors, useful_mask, anchor_mean_std = self.anchors(img_batch, P2, is_filtering=is_filtering)
        return_dict=dict(
            anchors=anchors, #[1, N, 4]
            mask=useful_mask, #[B, N]
            anchor_mean_std_3d=anchor_mean_std  #[N, C, K=6, 2]
        )
        return return_dict

    def _get_anchor_3d(self, anchors, anchor_mean_std_3d, assigned_labels):
        """
            anchors: [N_pos, 4] only positive anchors
            anchor_mean_std_3d: [N_pos, C, K=6, 2]
            assigned_labels: torch.Longtensor [N_pos, ]
            return:
                selected_mask = torch.bool [N_pos, ]
                selected_anchors_3d:  [N_selected, K, 2]
        """
        one_hot_mask = tf.cast(tf.one_hot(assigned_labels, self.num_classes), tf.bool)
        selected_anchor_3d = anchor_mean_std_3d[one_hot_mask]

        selected_mask = selected_anchor_3d[:, 0, 0] > 0 #only z > 0, filter out anchors with good variance and mean
        selected_anchor_3d = selected_anchor_3d[selected_mask]

        return selected_mask, selected_anchor_3d

    def get_bboxes(self, cls_scores, reg_preds, anchors, P2s, img_batch=None):
        
        assert cls_scores.shape[0] == 1 # batch == 1
        cls_scores = tf.math.sigmoid(cls_scores)

        cls_score = cls_scores[0][..., 0:self.num_classes]
        alpha_score = cls_scores[0][..., self.num_classes:self.num_classes+1]
        reg_pred  = reg_preds[0]
        
        anchor = anchors['anchors'][0] #[N, 4]
        anchor_mean_std_3d = anchors['anchor_mean_std_3d'] #[N, K, 2]
        useful_mask = anchors['mask'][0] #[N, ]

        anchor = anchor[useful_mask]
        cls_score = cls_score[useful_mask]
        alpha_score = alpha_score[useful_mask]
        reg_pred = reg_pred[useful_mask]
        anchor_mean_std_3d = anchor_mean_std_3d[useful_mask] #[N, K, 2]

        score_thr = getattr(self.test_cfg, 'score_thr', 0.5)
        # max_score, label = cls_score.max(dim=-1) 
        max_score = tf.reduce_max(cls_score, axis=-1)
        label = tf.argmax(cls_score, axis=-1)

        high_score_mask = (max_score > score_thr)
        high_score_inds = tf.squeeze(tf.where(high_score_mask), axis=1)

        # anchor      = anchor[high_score_mask, :]
        # anchor_mean_std_3d = anchor_mean_std_3d[high_score_mask, :]
        # cls_score   = cls_score[high_score_mask, :]
        # alpha_score = alpha_score[high_score_mask, :]
        # reg_pred    = reg_pred[high_score_mask, :]
        # max_score   = max_score[high_score_mask]
        # label       = label[high_score_mask]
        anchor = tf.gather(anchor, high_score_inds)
        anchor_mean_std_3d = tf.gather(anchor_mean_std_3d, high_score_inds)
        cls_score = tf.gather(cls_score, high_score_inds)
        alpha_score = tf.gather(alpha_score, high_score_inds)
        reg_pred = tf.gather(reg_pred, high_score_inds)
        max_score = tf.gather(max_score, high_score_inds)
        label = tf.gather(label, high_score_inds)

        bboxes, mask = self._decode(anchor, reg_pred, anchor_mean_std_3d, label, alpha_score)
        if img_batch is not None:
            bboxes = self.clipper(bboxes, img_batch)
        cls_score = cls_score[mask]
        max_score = max_score[mask]

        cls_agnostic = getattr(self.test_cfg, 'cls_agnositc', True) # True -> directly NMS; False -> NMS with offsets different categories will not collide
        nms_iou_thr  = getattr(self.test_cfg, 'nms_iou_thr', 0.5)
        
        if cls_agnostic:
            # keep_inds = nms(bboxes[:, :4], max_score, nms_iou_thr)
            keep_inds = tf.image.non_max_suppression(
                bboxes[:, :4], max_score, max_output_size=200, iou_threshold=nms_iou_thr
            )
        else:
            # max_coordinate = bboxes.max()
            max_coordinate = tf.reduce_max(bboxes)
            nms_bbox = bboxes[:, :4] + tf.expand_dims(tf.cast(label, tf.float32), 0) * max_coordinate
            # keep_inds = nms(nms_bbox, max_score, nms_iou_thr)
            keep_inds = tf.image.non_max_suppression(
                nms_bbox, max_score, max_output_size=200, iou_threshold=nms_iou_thr
            )

        # bboxes      = bboxes[keep_inds]
        # max_score   = max_score[keep_inds]
        # label       = label[keep_inds]
        bboxes = tf.gather(bboxes, keep_inds)
        max_score = tf.gather(max_score, keep_inds)
        label = tf.gather(label, keep_inds)

        is_post_opt = getattr(self.test_cfg, 'post_optimization', False)
        if is_post_opt:
            max_score, bboxes, label = self._post_process(max_score, bboxes, label, P2s)

        return max_score, bboxes, label

    def loss(self, cls_scores, reg_preds, anchors, annotations, P2s):
        batch_size = cls_scores.shape[0]

        anchor = anchors['anchors'][0] #[N, 4]
        anchor_mean_std_3d = anchors['anchor_mean_std_3d']
        cls_loss = []
        reg_loss = []
        number_of_positives = []
        for j in range(batch_size):
            
            reg_pred  = reg_preds[j]
            cls_score = cls_scores[j][..., 0:self.num_classes]
            alpha_score = cls_scores[j][..., self.num_classes:self.num_classes+1]

            # selected by mask
            useful_mask = anchors['mask'][j] #[N]
            anchor_j = anchor[useful_mask]
            anchor_mean_std_3d_j = anchor_mean_std_3d[useful_mask]
            reg_pred = reg_pred[useful_mask]
            cls_score = cls_score[useful_mask]
            alpha_score = alpha_score[useful_mask]

            # only select useful bbox_annotations
            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]#[k]

            if len(bbox_annotation) == 0:
                cls_loss.append(tf.constant(0., dtype=tf.float32))
                reg_loss.append(tf.zeros(self.num_regression_loss_terms, dtype=tf.float32))
                number_of_positives.append(0)
                continue

            assignement_result_dict = self._assign(anchor_j, bbox_annotation, **self.loss_cfg)
            sampling_result_dict    = self._sample(assignement_result_dict, anchor_j, bbox_annotation)
        
            num_valid_anchors = anchor_j.shape[0]
            labels = tf.fill(
                (num_valid_anchors, self.num_classes),
                -1,
            )

            pos_inds = sampling_result_dict['pos_inds']
            neg_inds = sampling_result_dict['neg_inds']
            
            if len(pos_inds) > 0:
                # pos_assigned_gt_label = bbox_annotation[sampling_result_dict['pos_assigned_gt_inds'], 4].long()
                pos_assigned_gt_label = tf.cast(tf.gather(bbox_annotation[:, 4], sampling_result_dict['pos_assigned_gt_inds']), tf.int32)
                
                selected_mask, selected_anchor_3d = self._get_anchor_3d(
                    sampling_result_dict['pos_bboxes'],
                    # anchor_mean_std_3d_j[pos_inds],
                    tf.gather(anchor_mean_std_3d_j, pos_inds),
                    pos_assigned_gt_label,
                )
                if len(selected_anchor_3d) > 0:
                    pos_inds = pos_inds[selected_mask]
                    pos_bboxes    = sampling_result_dict['pos_bboxes'][selected_mask]
                    pos_gt_bboxes = sampling_result_dict['pos_gt_bboxes'][selected_mask]
                    pos_assigned_gt = sampling_result_dict['pos_assigned_gt_inds'][selected_mask]

                    pos_bbox_targets, targets_alpha_cls = self._encode(
                        pos_bboxes, pos_gt_bboxes, selected_anchor_3d
                    ) #[N, 12], [N, 1]
                    label_index = pos_assigned_gt_label[selected_mask]

                    # Updates
                    inds = tf.reshape(pos_inds, (-1, 1))
                    updates = tf.zeros((inds.shape[0], labels.shape[1]), dtype=labels.dtype)
                    labels = tf.tensor_scatter_nd_update(labels, inds, updates)
                    inds = tf.stack([pos_inds, label_index], axis=1)
                    updates = tf.ones(inds.shape[0], dtype=labels.dtype)
                    labels = tf.tensor_scatter_nd_update(labels, inds, updates)

                    pos_anchor = tf.gather(anchor, pos_inds)
                    pos_alpha_score = tf.gather(alpha_score, pos_inds)
                    if self.decode_before_loss:
                        # TODO: What is ``anchors_3d_mean_std`?
                        # pos_prediction_decoded = self._decode(pos_anchor, reg_pred[pos_inds],  anchors_3d_mean_std, label_index, pos_alpha_score)
                        # pos_target_decoded     = self._decode(pos_anchor, pos_bbox_targets,  anchors_3d_mean_std, label_index, pos_alpha_score)

                        # reg_loss.append((self.loss_bbox(pos_prediction_decoded, pos_target_decoded)* self.regression_weight).mean(dim=0))
                        pass
                    else:
                        reg_loss_j = self.loss_bbox(pos_bbox_targets, tf.gather(reg_pred, pos_inds)) 
                        alpha_loss_j = self.alpha_loss(targets_alpha_cls, pos_alpha_score)
                        alpha_loss_j = tf.reshape(alpha_loss_j, pos_alpha_score.shape)
                        loss_j = tf.concat([reg_loss_j, alpha_loss_j], axis=1) * self.regression_weight #[N, 13]
                        reg_loss.append(tf.reduce_mean(loss_j, axis=0))
                        number_of_positives.append(bbox_annotation.shape[0])
            else:
                reg_loss.append(tf.zeros(self.num_regression_loss_terms, dtype=reg_preds.dtype))
                number_of_positives.append(bbox_annotation.shape[0])

            if len(neg_inds) > 0:
                inds = tf.reshape(neg_inds, (-1, 1))
                updates = tf.zeros((neg_inds.shape[0], labels.shape[1]), dtype=inds.dtype)
                labels = tf.tensor_scatter_nd_update(labels, inds, updates)
            
            cls_loss.append(tf.reduce_sum(self.loss_cls(cls_score, labels)) / (pos_inds.shape[0] + neg_inds.shape[0]))
        
        weights = tf.expand_dims(tf.constant(number_of_positives, dtype=reg_pred.dtype), axis=1)
        cls_loss = tf.reduce_mean(tf.stack(cls_loss), axis=0, keepdims=True)
        reg_loss = tf.stack(reg_loss, axis=0)

        weighted_regression_losses = tf.reduce_sum(
            weights * reg_loss / (tf.reduce_sum(weights) + 1e-6),
            axis=0
        )
        reg_loss = tf.reduce_mean(weighted_regression_losses, axis=0, keepdims=True)

        return cls_loss, reg_loss, dict(cls_loss=cls_loss, reg_loss=reg_loss, total_loss=cls_loss + reg_loss)


class StereoHead(AnchorBasedDetection3DHead):
    def init_layers(self,
                    num_anchors:int,
                    num_cls_output:int,
                    num_reg_output:int,
                    cls_feature_size:int=1024,
                    reg_feature_size:int=1024,
                    **kwargs):

        self.cls_feature_extraction = keras.Sequential([
            conv3x3(cls_feature_size, padding=1),
            layers.Dropout(0.3),
            layers.ReLU(),
            conv3x3(cls_feature_size, padding=1),
            layers.Dropout(0.3),
            layers.ReLU(),
            
            conv3x3(num_anchors*num_cls_output,
                    kernel_initializer=keras.initializers.Zeros(),
                    bias_initializer=keras.initializers.Zeros(),
                    padding=1),
            AnchorFlatten(num_cls_output),
        ])

        self.reg_feature_extraction = keras.Sequential([
            conv_bn_relu(reg_feature_size, 3),
            BasicBlock(reg_feature_size),
            layers.ReLU(),
            conv3x3(num_anchors*num_reg_output,
                    kernel_initializer=keras.initializers.Zeros(),
                    bias_initializer=keras.initializers.Zeros(),
                    padding=1),
            AnchorFlatten(num_reg_output),
        ])