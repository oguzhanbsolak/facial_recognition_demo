###################################################################################################
#
# Copyright (C) 2022 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
MobileNet-v2 [1] network implementation for MAX78002.

[1] MobileNetV2: Inverted Residuals and Linear Bottlenecks, Sandler M, Howard A, Zhu M,
Zhmoginov A, Chen LC. arXiv preprint. arXiv:1801.04381, 2018.
"""
from torch import nn

import ai85.ai8x as ai8x
import ai85.ai8x_blocks as ai8x_blocks
import torch.nn.functional as F

class AI87MobileNetV2(nn.Module):
    """
    MobileNet v2 for MAX78002
    """

    def __init__(  # pylint: disable=too-many-arguments
            self,
            pre_layer_stride,
            bottleneck_settings,
            last_layer_width,
            avg_pool_size=4,
            num_classes=100,
            num_channels=3,
            dimensions=(112, 112),  # pylint: disable=unused-argument
            bias=False,
            depthwise_bias=False,
            reduced_depthwise_bias=False,
            wide = True,
            **kwargs
    ):
        super().__init__()

        self.pre_stage = ai8x.FusedConv2dReLU(num_channels, bottleneck_settings[0][1], 3,
                                                padding=1, stride=pre_layer_stride,
                                                bias=False, **kwargs) # bias = False as we may need streaming
        self.dwise = ai8x.FusedMaxPoolDepthwiseConv2dReLU(64, 64, 3, padding=1, stride=1,
                                                            pool_size=2, pool_stride=2,
                                                             bias=depthwise_bias, **kwargs)
        self.feature_stage = nn.ModuleList([])
        for setting in bottleneck_settings:
            self._create_bottleneck_stage(setting, bias, depthwise_bias, reduced_depthwise_bias, **kwargs)

        self.post_stage = ai8x.FusedConv2dReLU(bottleneck_settings[-1][2], last_layer_width, 1,
                                               padding=0, stride=1, bias=False, **kwargs)

        self.classifier = ai8x.FusedAvgPoolConv2d(last_layer_width, 64, 1, padding=0,
                                                  stride=1, pool_size=avg_pool_size,
                                                  pool_stride=1, bias=False, wide=wide,
                                                  **kwargs)

    def _create_bottleneck_stage(self, setting, bias, depthwise_bias, reduced_depthwise_bias, **kwargs):
        """Function to create bottlencek stage. Setting format is:
           [num_repeat, in_channels, out_channels, stride, expansion_factor]
        """
        stage = []

        if setting[0] > 0:
            stage.append(ai8x_blocks.ResidualBottleneck(in_channels=setting[1],
                                                        out_channels=setting[2],
                                                        stride=setting[3],
                                                        expansion_factor=setting[4],
                                                        bias=bias, depthwise_bias=depthwise_bias,
                                                        **kwargs))

            for i in range(1, setting[0]):
                if reduced_depthwise_bias:
                    stage.append(ai8x_blocks.ResidualBottleneck(in_channels=setting[2],
                                                                out_channels=setting[2],
                                                                stride=1,
                                                                expansion_factor=setting[4],
                                                                bias=bias,
                                                                depthwise_bias=(i%2==0) and depthwise_bias, #Reduce depthwise bias
                                                                **kwargs))
                else:
                    stage.append(ai8x_blocks.ResidualBottleneck(in_channels=setting[2],
                                                                out_channels=setting[2],
                                                                stride=1,
                                                                expansion_factor=setting[4],
                                                                bias=bias,
                                                                depthwise_bias=depthwise_bias,
                                                                **kwargs))

        self.feature_stage.append(nn.Sequential(*stage))

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.pre_stage(x)
        x = self.dwise(x)
        for stage in self.feature_stage:
            x = stage(x)
        x = self.post_stage(x)
        x = self.classifier(x)
        x = F.normalize(x, p=2, dim=1)
        return x

def ai87netmobilefacenet_112(pretrained=False, **kwargs):
    
    assert not pretrained
    # settings for bottleneck stages in format
    # [num_repeat, in_channels, out_channels, stride, expansion_factor]
    bottleneck_settings = [
        [5, 64, 64, 2, 2],
        [1, 64, 128, 2, 4],
        [6, 128, 128, 1, 2],
        [1, 128, 128, 2, 4],
        [2, 128, 128, 1, 2]
    ]

    return AI87MobileNetV2(pre_layer_stride=1, bottleneck_settings=bottleneck_settings,
                           last_layer_width=128, avg_pool_size=(7,7),  depthwise_bias=True, reduced_depthwise_bias = True, wide=False, **kwargs) #Dwsize bias is false due to bias memory constraints
models = [
    {
        'name': 'ai87netmobilefacenet_112',
        'min_input': 1,
        'dim': 3,
    },
]
