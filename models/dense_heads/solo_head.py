import numpy as np
from scipy import ndimage
import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init
from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_loss
from .base_dense_seg_head import BaseDenseSegHead

from ..utils import matrix_nms

INF = 1e8


def points_nms(heat, kernel=2):
    # kernel must be 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=1)
    keep = (hmax[:, :, :-1, :-1] == heat).float()
    return heat * keep


def dice_loss(input, target):
    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1).float()

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    return 1-d


@HEADS.register_module()
class SOLOHead(BaseDenseSegHead):
    """SOLO: Segmenting Objects by Locations
    https://arxiv.org/abs/1912.04488
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 seg_feat_channels=256,
                 stacked_convs=4,
                 strides=(8, 8, 16, 32, 32),
                 base_edge_list=(16, 32, 64, 128, 256),
                 scale_ranges=((1, 96), (48, 192), (96, 384),
                               (192, 768), (384, 2048)),
                 sigma=0.2,
                 num_grids=None,
                 cate_down_pos=0,
                 background_label=None,
                 loss_mask=None,
                 loss_cls=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 train_cfg=None,
                 test_cfg=None):
        super(SOLOHead, self).__init__()
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.sigma = sigma
        self.cate_down_pos = cate_down_pos
        self.base_edge_list = base_edge_list
        self.scale_ranges = scale_ranges
        self.background_label = (
            num_classes if background_label is None else background_label)
        # background_label should be either 0 or num_classes
        assert (self.background_label == 0
                or self.background_label == num_classes)
        self.loss_cls = build_loss(loss_cls)
        self.ins_loss_weight = loss_mask['loss_weight']
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers()

    def _init_layers(self):
        self.ins_convs = nn.ModuleList()
        self.cate_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels + 2 if i == 0 else self.seg_feat_channels
            self.ins_convs.append(
                ConvModule(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))

            chn = self.in_channels if i == 0 else self.seg_feat_channels
            self.cate_convs.append(
                ConvModule(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))

        self.solo_ins_list = nn.ModuleList()
        for seg_num_grid in self.seg_num_grids:
            self.solo_ins_list.append(
                nn.Conv2d(
                    self.seg_feat_channels, seg_num_grid**2, 1))

        self.solo_cate = nn.Conv2d(
            self.seg_feat_channels, self.cate_out_channels, 3, padding=1)

    def init_weights(self):
        for m in self.ins_convs:
            normal_init(m.conv, std=0.01)
        for m in self.cate_convs:
            normal_init(m.conv, std=0.01)
        bias_ins = bias_init_with_prob(0.01)
        for m in self.solo_ins_list:
            normal_init(m, std=0.01, bias=bias_ins)
        bias_cate = bias_init_with_prob(0.01)
        normal_init(self.solo_cate, std=0.01, bias=bias_cate)

    def forward(self, feats, eval=False):
        new_feats = self.split_feats(feats)
        featmap_sizes = [featmap.size()[-2:] for featmap in new_feats]
        upsampled_size = (featmap_sizes[0][0] * 2, featmap_sizes[0][1] * 2)
        ins_pred, cate_pred = multi_apply(self.forward_single,
                                          new_feats,
                                          list(range(len(self.seg_num_grids))),
                                          eval=eval,
                                          upsampled_size=upsampled_size)
        return ins_pred, cate_pred

    def split_feats(self, feats):
        return (F.interpolate(feats[0], scale_factor=0.5, mode='bilinear', recompute_scale_factor=True),
                feats[1],
                feats[2],
                feats[3],
                F.interpolate(feats[4], size=feats[3].shape[-2:],
                              mode='bilinear', recompute_scale_factor=True))

    def forward_single(self, x, idx, eval=False, upsampled_size=None):
        ins_feat = x
        cate_feat = x
        # ins branch
        # concat coord
        x_range = torch.linspace(-1, 1, ins_feat.shape[-1],
                                 device=ins_feat.device)
        y_range = torch.linspace(-1, 1, ins_feat.shape[-2],
                                 device=ins_feat.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([ins_feat.shape[0], 1, -1, -1])
        x = x.expand([ins_feat.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        ins_feat = torch.cat([ins_feat, coord_feat], 1)

        for i, ins_layer in enumerate(self.ins_convs):
            ins_feat = ins_layer(ins_feat)

        ins_feat = F.interpolate(
            ins_feat, scale_factor=2, mode='bilinear', recompute_scale_factor=True)
        ins_pred = self.solo_ins_list[idx](ins_feat)

        # cate branch
        for i, cate_layer in enumerate(self.cate_convs):
            if i == self.cate_down_pos:
                seg_num_grid = self.seg_num_grids[idx]
                cate_feat = F.interpolate(cate_feat, size=seg_num_grid,
                                          mode='bilinear', recompute_scale_factor=True)
            cate_feat = cate_layer(cate_feat)

        cate_pred = self.solo_cate(cate_feat)
        if eval:
            ins_pred = F.interpolate(ins_pred.sigmoid(), size=upsampled_size,
                                     mode='bilinear', recompute_scale_factor=True)
            cate_pred = points_nms(cate_pred.sigmoid(),
                                   kernel=2).permute(0, 2, 3, 1)
        return ins_pred, cate_pred

    def loss(self,
             ins_preds,
             cate_preds,
             gt_bbox_list,
             gt_label_list,
             gt_mask_list,
             img_metas,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in
                         ins_preds]
        ins_label_list, cate_label_list, ins_ind_label_list = multi_apply(
            self.solo_target_single,
            gt_bbox_list,
            gt_label_list,
            gt_mask_list,
            featmap_sizes=featmap_sizes)

        # ins
        ins_labels = [torch.cat([ins_labels_level_img[ind_level_img, ...]
                                 for ins_labels_level_img, ind_level_img in
                                 zip(ins_labels_level, ins_ind_labels_level)], 0)
                      for ins_labels_level, ins_ind_labels_level in
                      zip(zip(*ins_label_list), zip(*ins_ind_label_list))]

        ins_preds = [torch.cat([ins_preds_level_img[ind_level_img, ...]
                                for ins_preds_level_img, ind_level_img in
                                zip(ins_preds_level, ins_ind_labels_level)], 0)
                     for ins_preds_level, ins_ind_labels_level in
                     zip(ins_preds, zip(*ins_ind_label_list))]

        ins_ind_labels = [
            torch.cat([ins_ind_labels_level_img.flatten()
                       for ins_ind_labels_level_img in ins_ind_labels_level])
            for ins_ind_labels_level in zip(*ins_ind_label_list)
        ]
        flatten_ins_ind_labels = torch.cat(ins_ind_labels)

        num_ins = flatten_ins_ind_labels.sum()

        # cate
        cate_labels = [
            torch.cat([cate_labels_level_img.flatten()
                       for cate_labels_level_img in cate_labels_level])
            for cate_labels_level in zip(*cate_label_list)
        ]
        flatten_cate_labels = torch.cat(cate_labels)
        cate_preds = [
            cate_pred.permute(0, 2, 3, 1).reshape(-1, self.cate_out_channels)
            for cate_pred in cate_preds
        ]
        flatten_cate_preds = torch.cat(cate_preds)
        loss_cls = self.loss_cls(flatten_cate_preds, flatten_cate_labels,
                                 avg_factor=num_ins + 1)

        # dice loss
        loss_mask = []
        for input, target in zip(ins_preds, ins_labels):
            if input.size()[0] == 0:
                continue
            input = torch.sigmoid(input)
            loss_mask.append(dice_loss(input, target))
        loss_mask = torch.cat(loss_mask).mean()
        loss_mask = loss_mask * self.ins_loss_weight
        return dict(
            loss_mask=loss_mask,
            loss_cls=loss_cls)

    def solo_target_single(self,
                           gt_bboxes_raw,
                           gt_labels_raw,
                           gt_masks_raw,
                           featmap_sizes=None):

        device = gt_labels_raw[0].device

        # ins
        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) *
                              (gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))

        ins_label_list = []
        cate_label_list = []
        ins_ind_label_list = []
        for (lower_bound, upper_bound), stride, featmap_size, num_grid \
            in zip(self.scale_ranges, self.strides,
                   featmap_sizes, self.seg_num_grids):

            ins_label = torch.zeros([num_grid ** 2,
                                     featmap_size[0], featmap_size[1]],
                                    dtype=torch.uint8, device=device)
            # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
            cate_label = torch.zeros([num_grid, num_grid],
                                     dtype=torch.int64,
                                     device=device) + self.num_classes
            ins_ind_label = torch.zeros([num_grid ** 2],
                                        dtype=torch.bool, device=device)

            hit_indices = ((gt_areas >= lower_bound) &
                           (gt_areas <= upper_bound)).nonzero(as_tuple=False).flatten()
            if len(hit_indices) == 0:
                ins_label_list.append(ins_label)
                cate_label_list.append(cate_label)
                ins_ind_label_list.append(ins_ind_label)
                continue
            gt_bboxes = gt_bboxes_raw[hit_indices]
            gt_labels = gt_labels_raw[hit_indices]
            gt_masks = gt_masks_raw[hit_indices.cpu().numpy(), ...]

            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

            output_stride = stride / 2

            for seg_mask, gt_label, half_h, half_w in \
                    zip(gt_masks, gt_labels, half_hs, half_ws):
                if seg_mask.sum() < 10:
                    continue
                # mass center
                upsampled_size = (featmap_sizes[0][0] * 4,
                                  featmap_sizes[0][1] * 4)
                center_h, center_w = ndimage.measurements.center_of_mass(
                    seg_mask)
                coord_w = int((center_w / upsampled_size[1]
                               ) // (1. / num_grid))
                coord_h = int((center_h / upsampled_size[0]
                               ) // (1. / num_grid))

                # left, top, right, down
                top_box = max(0,
                              int(((center_h - half_h) /
                                   upsampled_size[0]) // (1. / num_grid)))
                down_box = min(num_grid - 1,
                               int(((center_h + half_h) /
                                    upsampled_size[0]) // (1. / num_grid)))
                left_box = max(0,
                               int(((center_w - half_w) /
                                    upsampled_size[1]) // (1. / num_grid)))
                right_box = min(num_grid - 1,
                                int(((center_w + half_w) /
                                     upsampled_size[1]) // (1. / num_grid)))

                top = max(top_box, coord_h-1)
                down = min(down_box, coord_h+1)
                left = max(coord_w-1, left_box)
                right = min(right_box, coord_w+1)

                cate_label[top:(down+1), left:(right+1)] = gt_label
                # ins
                seg_mask = mmcv.imrescale(seg_mask, scale=1. / output_stride)
                seg_mask = torch.Tensor(seg_mask)
                for i in range(top, down+1):
                    for j in range(left, right+1):
                        label = int(i * num_grid + j)
                        ins_label[label, :seg_mask.shape[0],
                                  :seg_mask.shape[1]] = seg_mask
                        ins_ind_label[label] = True
            ins_label_list.append(ins_label)
            cate_label_list.append(cate_label)
            ins_ind_label_list.append(ins_ind_label)
        return ins_label_list, cate_label_list, ins_ind_label_list

    def get_seg(self, seg_preds, cate_preds, img_metas, cfg, rescale=None):
        assert len(seg_preds) == len(cate_preds)
        num_levels = len(cate_preds)
        featmap_size = seg_preds[0].size()[-2:]

        bbox_result_list = []
        segm_result_list = []
        for img_id in range(len(img_metas)):
            cate_pred_list = [
                cate_preds[i][img_id].view(
                    -1, self.cate_out_channels).detach() for i in
                range(num_levels)
            ]
            seg_pred_list = [
                seg_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            ori_shape = img_metas[img_id]['ori_shape']

            cate_pred_list = torch.cat(cate_pred_list, dim=0)
            seg_pred_list = torch.cat(seg_pred_list, dim=0)

            result = self.get_seg_single(cate_pred_list, seg_pred_list,
                                         featmap_size, img_shape, ori_shape,
                                         scale_factor, cfg, rescale)
            bbox_result, segm_result = self.segm2result(result)
            bbox_result_list.append(bbox_result)
            segm_result_list.append(segm_result)
        return bbox_result_list, segm_result_list

    def segm2result(self, result):
        if result is None:
            bbox_result = [np.zeros((0, 5), dtype=np.float32) for i in
                           range(self.num_classes)]
            # BG is not included in num_classes
            segm_result = [[] for _ in range(self.num_classes)]
        else:
            bbox_result = [np.zeros((0, 5), dtype=np.float32) for i in
                           range(self.num_classes)]
            segm_result = [[] for _ in range(self.num_classes)]
            seg_pred = result[0].detach().cpu().numpy()
            cate_label = result[1].detach().cpu().numpy()
            cate_score = result[2].detach().cpu().numpy()
            num_ins = seg_pred.shape[0]
            # extract bboxes from segmentation result
            bboxes = np.zeros((num_ins, 5), dtype=np.float32)
            bboxes[:, -1] = cate_score
            bboxes[:, :-1] = self.extract_bboxes(seg_pred)

            bbox_result = [bboxes[cate_label == i, :] for i in
                           range(self.num_classes)]

            for idx in range(num_ins):
                segm_result[cate_label[idx]].append(seg_pred[idx])
        return bbox_result, segm_result

    def extract_bboxes(self, mask):
        """Compute bounding boxes from masks.
        mask: [num_instances, height, width]. Mask pixels are either 1 or 0.
        Returns: bbox array [num_instances, (x1, y1, x2, y2)].
        """
        num_ins = mask.shape[0]
        boxes = np.zeros([num_ins, 4], dtype=np.float32)
        for i in range(num_ins):
            m = mask[i, :, :]
            # Bounding box.
            horizontal_indicies = np.where(np.any(m, axis=0))[0]
            vertical_indicies = np.where(np.any(m, axis=1))[0]
            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                # x2 and y2 should not be part of the box. Increment by 1.
                x2 += 1
                y2 += 1
            else:
                # No mask for this instance. Might happen due to
                # resizing or cropping. Set bbox to zeros
                x1, x2, y1, y2 = 0, 0, 0, 0
            boxes[i] = np.array([x1, y1, x2, y2])
        return boxes

    def get_seg_single(self,
                       cate_preds,
                       seg_preds,
                       featmap_size,
                       img_shape,
                       ori_shape,
                       scale_factor,
                       cfg,
                       rescale=False):
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cate_preds) == len(seg_preds)

        # overall info.
        h, w, _ = img_shape
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)

        # process.
        inds = (cate_preds > cfg.score_thr)
        # category scores.
        cate_scores = cate_preds[inds]
        if len(cate_scores) == 0:
            return None
        # category labels.
        inds = inds.nonzero(as_tuple=False)
        cate_labels = inds[:, 1]

        # strides.
        size_trans = cate_labels.new_tensor(
            self.seg_num_grids).pow(2).cumsum(0)
        strides = cate_scores.new_ones(size_trans[-1])
        n_stage = len(self.seg_num_grids)
        strides[:size_trans[0]] *= self.strides[0]
        for ind_ in range(1, n_stage):
            strides[size_trans[ind_ - 1]:size_trans[ind_]] *= \
                self.strides[ind_]
        strides = strides[inds[:, 0]]

        # masks.
        seg_preds = seg_preds[inds[:, 0]]
        seg_masks = seg_preds > cfg.mask_thr
        sum_masks = seg_masks.sum((1, 2)).float()

        # filter.
        keep = sum_masks > strides
        if keep.sum() == 0:
            return None

        seg_masks = seg_masks[keep, ...]
        seg_preds = seg_preds[keep, ...]
        sum_masks = sum_masks[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # maskness.
        seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_scores

        # sort and keep top nms_pre
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg.nms_pre:
            sort_inds = sort_inds[:cfg.nms_pre]
        seg_masks = seg_masks[sort_inds, :, :]
        seg_preds = seg_preds[sort_inds, :, :]
        sum_masks = sum_masks[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        # Matrix NMS
        cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores,
                                 kernel=cfg.kernel, sigma=cfg.sigma,
                                 sum_masks=sum_masks)

        # filter.
        keep = cate_scores >= cfg.update_thr
        if keep.sum() == 0:
            return None
        seg_preds = seg_preds[keep, :, :]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # sort and keep top_k
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg.max_per_img:
            sort_inds = sort_inds[:cfg.max_per_img]
        seg_preds = seg_preds[sort_inds, :, :]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        seg_preds = F.interpolate(seg_preds.unsqueeze(0),
                                  size=upsampled_size_out,
                                  mode='bilinear', recompute_scale_factor=True)[:, :, :h, :w]
        seg_masks = F.interpolate(seg_preds,
                                  size=ori_shape[:2],
                                  mode='bilinear', recompute_scale_factor=True).squeeze(0)
        seg_masks = seg_masks > cfg.mask_thr
        return seg_masks, cate_labels, cate_scores