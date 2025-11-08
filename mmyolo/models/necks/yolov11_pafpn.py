# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import torch.nn as nn
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.registry import MODELS
from .. import C2fCIB
from ..utils import make_divisible, make_round
from .yolov8_pafpn import YOLOv8PAFPN


@MODELS.register_module()
class YOLOv11PAFPN(YOLOv8PAFPN):
    """PAFPN for YOLOv11 models.

    It reuses the overall topology of :class:`YOLOv8PAFPN` but swaps the
    ``CSPLayerWithTwoConv`` blocks with the new :class:`C2fCIB` blocks that are
    aligned with the YOLOv11 backbone design.
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 freeze_all: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None,
                 c2f_expand_ratio: float = 0.5,
                 cib_expand_ratio: float = 0.5):
        self.c2f_expand_ratio = c2f_expand_ratio
        self.cib_expand_ratio = cib_expand_ratio
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            num_csp_blocks=num_csp_blocks,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)

    def build_reduce_layer(self, idx: int) -> nn.Module:
        # Same identity reduce layer as YOLOv8.
        return nn.Identity()

    def _build_c2f_cib(self, in_channels: int, out_channels: int,
                       num_blocks: int) -> nn.Module:
        return C2fCIB(
            in_channels,
            out_channels,
            expand_ratio=self.c2f_expand_ratio,
            num_blocks=num_blocks,
            add_identity=False,
            cib_expand_ratio=self.cib_expand_ratio,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_top_down_layer(self, idx: int) -> nn.Module:
        return self._build_c2f_cib(
            make_divisible(self.in_channels[idx - 1] + self.in_channels[idx],
                           self.widen_factor),
            make_divisible(self.out_channels[idx - 1], self.widen_factor),
            make_round(self.num_csp_blocks, self.deepen_factor))

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        return self._build_c2f_cib(
            make_divisible(self.out_channels[idx] + self.out_channels[idx + 1],
                           self.widen_factor),
            make_divisible(self.out_channels[idx + 1], self.widen_factor),
            make_round(self.num_csp_blocks, self.deepen_factor))
