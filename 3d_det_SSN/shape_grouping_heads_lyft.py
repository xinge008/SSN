
# -*- coding:utf-8 -*-


import time
from enum import Enum
from functools import reduce
import contextlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from second.pytorch.models.voxelnet import register_voxelnet, VoxelNet
from second.pytorch.models import rpn
from .shape_aware_multi_heads_lyft import SmallObjectHead, HugeHead, LargeHead, TinyObjectHead
# num_class_attr = 9


@register_voxelnet
class Shape_Grouping_Heads_lyft(VoxelNet):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        assert self._num_class == 9
        assert isinstance(self.rpn, rpn.RPNNoHead)
        self.small_classes = ["bicycle", "motorcycle"]
        self.tiny_classes = ["pedestrian", "animal"]
        self.large_classes = ["car", "emergency_vehicle"]
        self.huge_classes = ["bus", "other_vehicle", "truck"]
        small_num_anchor_loc = sum([self.target_assigner.num_anchors_per_location_class(c) for c in self.small_classes])
        tiny_num_anchor_loc = sum([self.target_assigner.num_anchors_per_location_class(c) for c in self.tiny_classes])
        large_num_anchor_loc = sum([self.target_assigner.num_anchors_per_location_class(c) for c in self.large_classes])
        huge_num_anchor_loc = sum([self.target_assigner.num_anchors_per_location_class(c) for c in self.huge_classes])

        """
        small-object: 248x248  -->  0.4x0.4
        large-object: 124x124  -->  0.8x0.8
        """

        self.small_head = SmallObjectHead(
            num_filters=np.sum(self.rpn._num_upsample_filters),
            num_class=self._num_class,
            num_anchor_per_loc=small_num_anchor_loc,
            encode_background_as_zeros=self._encode_background_as_zeros,
            use_direction_classifier=self._use_direction_classifier,
            box_code_size=self._box_coder.code_size,
            num_direction_bins=self._num_direction_bins,
        )
        self.tiny_head = TinyObjectHead(
            num_filters=np.sum(self.rpn._num_upsample_filters),
            num_class=self._num_class,
            num_anchor_per_loc=tiny_num_anchor_loc,
            encode_background_as_zeros=self._encode_background_as_zeros,
            use_direction_classifier=self._use_direction_classifier,
            box_code_size=self._box_coder.code_size,
            num_direction_bins=self._num_direction_bins,
        )

        self.large_head = LargeHead(
            num_filters=np.sum(self.rpn._num_upsample_filters),
            num_class=self._num_class,
            num_anchor_per_loc=large_num_anchor_loc,
            encode_background_as_zeros=self._encode_background_as_zeros,
            use_direction_classifier=self._use_direction_classifier,
            box_code_size=self._box_coder.code_size,
            num_direction_bins=self._num_direction_bins,
        )
        self.huge_head = HugeHead(
            num_filters=np.sum(self.rpn._num_upsample_filters),
            num_class=self._num_class,
            num_anchor_per_loc=huge_num_anchor_loc,
            encode_background_as_zeros=self._encode_background_as_zeros,
            use_direction_classifier=self._use_direction_classifier,
            box_code_size=self._box_coder.code_size,
            num_direction_bins=self._num_direction_bins,
        )

    def network_forward(self, voxels, num_points, coors, batch_size):
        self.start_timer("voxel_feature_extractor")
        voxel_features = self.voxel_feature_extractor(voxels, num_points,
                                                      coors)
        self.end_timer("voxel_feature_extractor")

        self.start_timer("middle forward")
        spatial_features = self.middle_feature_extractor(
            voxel_features, coors, batch_size)
        self.end_timer("middle forward")
        self.start_timer("rpn forward")
        """
        spatial_features: 1, 64, 496, 496
        """
        rpn_out = self.rpn(spatial_features)
        # r1 = rpn_out["stage0"]
        # _, _, H, W = r1.shape
        # cropsize40x40 = np.round(H * 0.1).astype(np.int64)
        # r1 = r1[:, :, cropsize40x40:-cropsize40x40, cropsize40x40:-cropsize40x40]
        """
        ['up0', 'up1', 'up2', 'stage0', 'stage1', 'stage2', 'out']
        out: 248x248

        """
        small = self.small_head(rpn_out["out"])
        tiny = self.tiny_head(rpn_out["out"])
        large = self.large_head(rpn_out["out"])
        huge = self.huge_head(rpn_out["out"])
        self.end_timer("rpn forward")
        # concated preds MUST match order in class_settings in config.
        res = {
            "box_preds": torch.cat([large["box_preds"], huge["box_preds"], small["box_preds"], tiny["box_preds"]], dim=1),
            "cls_preds": torch.cat([large["cls_preds"], huge["cls_preds"], small["cls_preds"], tiny["cls_preds"]], dim=1),
        }
        if self._use_direction_classifier:
            res["dir_cls_preds"] = torch.cat([large["dir_cls_preds"], huge["dir_cls_preds"], small["dir_cls_preds"],
                                              tiny["dir_cls_preds"]], dim=1)
        return res

