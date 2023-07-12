###################################################################################################
#
# Copyright (C) 2022 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################

from torch import nn

import ai85.ai8x as ai8x
import torch.nn.functional as F

class AI87DotProd(nn.Module):
    """
    Dotprod for MAX78002
    """

    def __init__(  # pylint: disable=too-many-arguments
            self,
            bias=False,
            **kwargs
    ):
        super().__init__()
        self.dot = ai8x.Linear(64, 1024, bias=False, **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.dot(x)
        return x

def dotprod(pretrained=False, **kwargs):
    
    assert not pretrained


    return AI87DotProd(bias = False, **kwargs) #Dwsize bias is false due to bias memory constraints
models = [
    {
        'name': 'dotprod',
        'min_input': 1,
        'dim': 3,
    },
]
