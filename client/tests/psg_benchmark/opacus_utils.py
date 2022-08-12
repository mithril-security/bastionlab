from typing import Iterator, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

def unfold2d(
    input,
    *,
    kernel_size: Tuple[int, int],
    padding: Tuple[int, int],
    stride: Tuple[int, int],
    dilation: Tuple[int, int],
):
    """
    See :meth:`~torch.nn.functional.unfold`
    """
    *shape, H, W = input.shape
    H_effective = (
        H + 2 * padding[0] - (kernel_size[0] + (kernel_size[0] - 1) * (dilation[0] - 1))
    ) // stride[0] + 1
    W_effective = (
        W + 2 * padding[1] - (kernel_size[1] + (kernel_size[1] - 1) * (dilation[1] - 1))
    ) // stride[1] + 1
    # F.pad's first argument is the padding of the *last* dimension
    input = F.pad(input, (padding[1], padding[1], padding[0], padding[0]))
    *shape_pad, H_pad, W_pad = input.shape
    strides = list(input.stride())
    strides = strides[:-2] + [
        W_pad * dilation[0],
        dilation[1],
        W_pad * stride[0],
        stride[1],
    ]
    out = input.as_strided(
        shape + [kernel_size[0], kernel_size[1], H_effective, W_effective], strides
    )

    return out.reshape(input.size(0), -1, H_effective * W_effective)