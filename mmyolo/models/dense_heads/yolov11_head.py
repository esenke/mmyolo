# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig

from mmyolo.registry import MODELS
from .yolov8_head import YOLOv8Head, YOLOv8HeadModule


@MODELS.register_module()
class YOLOv11HeadModule(YOLOv8HeadModule):
    """YOLOv11 head module with optional depthwise separable blocks."""

    def __init__(self,
                 num_classes: int,
                 in_channels: Union[int, Sequence],
                 widen_factor: float = 1.0,
                 num_base_priors: int = 1,
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 reg_max: int = 16,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None,
                 use_depthwise: bool = True,
                 stacked_convs: int = 2):
        self.use_depthwise = use_depthwise
        self.stacked_convs = stacked_convs
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            widen_factor=widen_factor,
            num_base_priors=num_base_priors,
            featmap_strides=featmap_strides,
            reg_max=reg_max,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)

    def _build_conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        if not self.use_depthwise:
            return nn.Sequential(
                ConvModule(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        return nn.Sequential(
            ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=out_channels,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))

    def _init_layers(self):
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()

        reg_out_channels = max(
            (16, self.in_channels[0] // 4, self.reg_max * 4))
        cls_out_channels = max(self.in_channels[0], self.num_classes)

        for i in range(self.num_levels):
            reg_layers: List[nn.Module] = []
            cls_layers: List[nn.Module] = []
            reg_in_c = self.in_channels[i]
            cls_in_c = self.in_channels[i]
            for _ in range(self.stacked_convs):
                reg_layers.append(
                    self._build_conv_block(reg_in_c, reg_out_channels))
                cls_layers.append(
                    self._build_conv_block(cls_in_c, cls_out_channels))
                reg_in_c = reg_out_channels
                cls_in_c = cls_out_channels
            reg_layers.append(
                nn.Conv2d(
                    in_channels=reg_out_channels,
                    out_channels=4 * self.reg_max,
                    kernel_size=1))
            cls_layers.append(
                nn.Conv2d(
                    in_channels=cls_out_channels,
                    out_channels=self.num_classes,
                    kernel_size=1))
            self.reg_preds.append(nn.Sequential(*reg_layers))
            self.cls_preds.append(nn.Sequential(*cls_layers))

        proj = torch.arange(self.reg_max, dtype=torch.float)
        self.register_buffer('proj', proj, persistent=False)


@MODELS.register_module()
class YOLOv11Head(YOLOv8Head):
    """Detection head for YOLOv11."""

    def __init__(self,
                 head_module: ConfigType,
                 prior_generator: ConfigType = dict(
                     type='mmdet.MlvlPointGenerator',
                     offset=0.5,
                     strides=[8, 16, 32]),
                 bbox_coder: ConfigType = dict(type='DistancePointBBoxCoder'),
                 loss_cls: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='none',
                     loss_weight=0.5),
                 loss_bbox: ConfigType = dict(
                     type='IoULoss',
                     iou_mode='ciou',
                     bbox_format='xyxy',
                     reduction='sum',
                     loss_weight=7.5,
                     return_iou=False),
                 loss_dfl=dict(
                     type='mmdet.DistributionFocalLoss',
                     reduction='mean',
                     loss_weight=1.5 / 4),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            head_module=head_module,
            prior_generator=prior_generator,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_dfl=loss_dfl,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)

    # The rest of the logic (loss computation, etc.) is inherited from YOLOv8.
