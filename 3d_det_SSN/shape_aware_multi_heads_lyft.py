import time
from enum import Enum
from functools import reduce
import contextlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# the shape vector and velocity vector are incorporated into the whole box_code
_box_code_size = 7+2+9 # box_code=7, velo_code=2, shape_code=9
_num_class = 9
num_class_attr = 9


class SmallObjectHead(nn.Module):
    def __init__(self, num_filters, num_class, num_anchor_per_loc,
                 box_code_size, num_direction_bins, use_direction_classifier,
                 encode_background_as_zeros):
        super().__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._num_direction_bins = num_direction_bins
        self._num_class = num_class
        self._use_direction_classifier = use_direction_classifier
        self._box_code_size = box_code_size
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
            # num_attr = num_anchor_per_loc * num_class_attr
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
            # num_attr = num_anchor_per_loc * (num_class_attr + 1)

        self.net = nn.Sequential(
            nn.Conv2d(num_filters, 64, 3, bias=False, padding=1, dilation=1),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, bias=False, padding=1),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.01),
            nn.ReLU(),
        )
        final_num_filters = 64
        self.conv_cls = nn.Conv2d(final_num_filters, num_cls, 1)
        self.conv_box = nn.Conv2d(final_num_filters,
                                  num_anchor_per_loc * _box_code_size, 1)

        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                final_num_filters, num_anchor_per_loc * num_direction_bins, 1)

        nn.init.constant_(self.conv_cls.bias, -np.log((1-0.01) / 0.01))
        # nn.init.constant_(self.conv_attr[-1].bias, 0)
        # nn.init.kaiming_normal_(self.conv_attr[-1].weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, idx):
        x = self.net(x)
        batch_size = x.shape[0]
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)

        C, H, W = box_preds.shape[1:]
        box_preds = box_preds.view(-1, self._num_anchor_per_loc,
                                   _box_code_size, H, W).permute(
                                       0, 1, 3, 4, 2).contiguous()

        cls_preds = cls_preds.view(-1, self._num_anchor_per_loc,
                                   self._num_class, H, W).permute(
                                       0, 1, 3, 4, 2).contiguous()



        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.view(
                -1, self._num_anchor_per_loc, self._num_direction_bins, H,
                W).permute(0, 1, 3, 4, 2).contiguous()

        # if idx == 2:

        box_preds = box_preds.view(batch_size, -1, _box_code_size)
        cls_preds = cls_preds.view(batch_size, -1, self._num_class)
        if self._use_direction_classifier:
            dir_cls_preds = dir_cls_preds.view(batch_size, -1, self._num_direction_bins)



        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }
        if self._use_direction_classifier:
            # dir_cls_preds = self.conv_dir_cls(x)

            # ret_dict["dir_cls_preds"] = dir_cls_preds.view(batch_size, -1, self._num_direction_bins)
            ret_dict["dir_cls_preds"] = dir_cls_preds

        return ret_dict

class TinyObjectHead(nn.Module):
    def __init__(self, num_filters, num_class, num_anchor_per_loc,
                 box_code_size, num_direction_bins, use_direction_classifier,
                 encode_background_as_zeros):
        super().__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._num_direction_bins = num_direction_bins
        self._num_class = num_class
        self._use_direction_classifier = use_direction_classifier
        self._box_code_size = box_code_size
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)

        self.net = nn.Sequential(
            nn.Conv2d(num_filters, 32, 3, bias=False, padding=1, dilation=1),
            nn.BatchNorm2d(32, eps=0.001, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, bias=False, padding=1),
            nn.BatchNorm2d(32, eps=0.001, momentum=0.01),
            nn.ReLU(),
        )

        final_num_filters = 32
        self.conv_cls = nn.Conv2d(final_num_filters, num_cls, 1)
        self.conv_box = nn.Conv2d(final_num_filters,
                                  num_anchor_per_loc * _box_code_size, 1)


        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                final_num_filters, num_anchor_per_loc * num_direction_bins, 1)

        nn.init.constant_(self.conv_cls.bias, -np.log((1-0.01) / 0.01))
        # nn.init.constant_(self.conv_attr[-1].bias, 0)
        # nn.init.kaiming_normal_(self.conv_attr[-1].weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):

        x = self.net(x)
        batch_size = x.shape[0]
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)

        # [N, C, y(H), x(W)]
        C, H, W = box_preds.shape[1:]
        box_preds = box_preds.view(-1, self._num_anchor_per_loc,
                                   _box_code_size, H, W).permute(
                                       0, 1, 3, 4, 2).contiguous()

        cls_preds = cls_preds.view(-1, self._num_anchor_per_loc,
                                   self._num_class, H, W).permute(
                                       0, 1, 3, 4, 2).contiguous()

        ret_dict = {
            "box_preds": box_preds.view(batch_size, -1, _box_code_size),
            "cls_preds": cls_preds.view(batch_size, -1, self._num_class),
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.view(
                -1, self._num_anchor_per_loc, self._num_direction_bins, H,
                W).permute(0, 1, 3, 4, 2).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds.view(batch_size, -1, self._num_direction_bins)
        return ret_dict

class LargeHead(nn.Module):
    def __init__(self, num_filters, num_class, num_anchor_per_loc,
                 box_code_size, num_direction_bins, use_direction_classifier,
                 encode_background_as_zeros):
        super().__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._num_direction_bins = num_direction_bins
        self._num_class = num_class
        self._use_direction_classifier = use_direction_classifier
        self._box_code_size = box_code_size
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)

        self.net = nn.Sequential(
            nn.Conv2d(num_filters, 64, 3, stride=2, bias=False, padding=1, dilation=1), # downsample
            nn.BatchNorm2d(64,eps=0.001, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, bias=False, padding=1),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, bias=False, padding=1, dilation=1),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.01),
            nn.ReLU(),
        )

        final_num_filters = 64

        self.conv_cls = nn.Conv2d(final_num_filters, num_cls, 1)
        self.conv_box = nn.Conv2d(final_num_filters,
                                  num_anchor_per_loc * _box_code_size, 1)

        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                final_num_filters, num_anchor_per_loc * num_direction_bins, 1)

        nn.init.constant_(self.conv_cls.bias, -np.log((1-0.01) / 0.01))
        # nn.init.kaiming_normal_(self.conv_attr[-1].bias, mode='fan_out', nonlinearity='relu')

    def forward(self, x, idx):
        x = self.net(x)
        batch_size = x.shape[0]
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)

        # [N, C, y(H), x(W)]
        C, H, W = box_preds.shape[1:]
        box_preds = box_preds.view(-1, self._num_anchor_per_loc,
                                   _box_code_size, H, W).permute(
                                       0, 1, 3, 4, 2).contiguous()

        cls_preds = cls_preds.view(-1, self._num_anchor_per_loc,
                                   self._num_class, H, W).permute(
                                       0, 1, 3, 4, 2).contiguous()



        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.view(
                -1, self._num_anchor_per_loc, self._num_direction_bins, H,
                W).permute(0, 1, 3, 4, 2).contiguous()

        # elif idx == 1:
        box_preds = box_preds.view(batch_size, -1, _box_code_size)
        cls_preds = cls_preds.view(batch_size, -1, self._num_class)
        if self._use_direction_classifier:
            dir_cls_preds = dir_cls_preds.view(batch_size, -1, self._num_direction_bins)



        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,

        }
        if self._use_direction_classifier:

            ret_dict["dir_cls_preds"] = dir_cls_preds #.view(batch_size, -1, self._num_direction_bins)
        return ret_dict

class HugeHead(nn.Module):
    def __init__(self, num_filters, num_class, num_anchor_per_loc,
                 box_code_size, num_direction_bins, use_direction_classifier,
                 encode_background_as_zeros):
        super().__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._num_direction_bins = num_direction_bins
        self._num_class = num_class
        self._use_direction_classifier = use_direction_classifier
        self._box_code_size = box_code_size
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)

        self.net = nn.Sequential(
            nn.Conv2d(num_filters, 128, 3, stride=2, bias=False, padding=1, dilation=1), # downsample
            nn.BatchNorm2d(128,eps=0.001, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, bias=False, padding=1, dilation=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, stride=2, bias=False, padding=1, dilation=1), # downsample
            nn.BatchNorm2d(64, eps=0.001, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, bias=False, padding=1, dilation=1),
            nn.BatchNorm2d(64,eps=0.001, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, bias=False, padding=1, dilation=1),
            nn.BatchNorm2d(64,eps=0.001, momentum=0.01),
            nn.ReLU(),
        )

        final_num_filters = 64

        self.conv_cls = nn.Conv2d(final_num_filters, num_cls, 1)
        self.conv_box = nn.Conv2d(final_num_filters,
                                  num_anchor_per_loc * _box_code_size, 1)


        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                final_num_filters, num_anchor_per_loc * num_direction_bins, 1)

        nn.init.constant_(self.conv_cls.bias, -np.log((1-0.01) / 0.01))

    def forward(self, x):
        x = self.net(x)
        batch_size = x.shape[0]
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)


        # [N, C, y(H), x(W)]
        C, H, W = box_preds.shape[1:]
        box_preds = box_preds.view(-1, self._num_anchor_per_loc,
                                   _box_code_size, H, W).permute(
                                       0, 1, 3, 4, 2).contiguous()

        cls_preds = cls_preds.view(-1, self._num_anchor_per_loc,
                                   self._num_class, H, W).permute(
                                       0, 1, 3, 4, 2).contiguous()

        ret_dict = {
            "box_preds": box_preds.view(batch_size, -1, _box_code_size),
            "cls_preds": cls_preds.view(batch_size, -1, self._num_class),
            # "attr_preds": attr_preds.view(batch_size, -1, num_class_attr),
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.view(
                -1, self._num_anchor_per_loc, self._num_direction_bins, H,
                W).permute(0, 1, 3, 4, 2).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds.view(batch_size, -1, self._num_direction_bins)
        return ret_dict
