import torch
from torch import Tensor
from torch.nn import Module, Conv2d, Parameter
from torch.nn.functional import conv2d
from opacus_utils import unfold2d
from private_module import PrivacyEngine
from typing import Tuple, Dict
import numpy as np
from time import time

N = 100

####################
# Method #1: Hooks #
####################

engine = PrivacyEngine()


@engine.grad_sample_fn(Conv2d)
def conv2d_grad_sample_fn(
    layer: Module, input: Tuple[Tensor], grad_output: Tuple[Tensor]
) -> Dict[Parameter, Tensor]:
    n = input[0].shape[0]
    # get activations and backprops in shape depending on the Conv layer
    if type(layer) == Conv2d:
        activations = unfold2d(
            input[0],
            kernel_size=layer.kernel_size,
            padding=layer.padding,
            stride=layer.stride,
            dilation=layer.dilation,
        )
    else:
        raise (Exception("Unimplemented"))
    backprops = grad_output[0].reshape(n, -1, activations.shape[-1])

    ret = {}
    if layer.weight.requires_grad:
        # n=batch_sz; o=num_out_channels; p=(num_in_channels/groups)*kernel_sz
        grad_sample = torch.einsum("noq,npq->nop", backprops, activations)
        # rearrange the above tensor and extract diagonals.
        grad_sample = grad_sample.view(
            n,
            layer.groups,
            -1,
            layer.groups,
            int(layer.in_channels / layer.groups),
            np.prod(layer.kernel_size),
        )
        grad_sample = torch.einsum("ngrg...->ngr...", grad_sample).contiguous()
        shape = [n] + list(layer.weight.shape)
        ret[layer.weight] = grad_sample.view(shape)

    if layer.bias is not None and layer.bias.requires_grad:
        ret[layer.bias] = torch.sum(backprops, dim=2)

    return ret


@engine.private_module(64)
class SingleConv(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(3, 16, 3)

    def forward(self, x):
        return self.conv1(x)


x = torch.randn(64, 3, 256, 256)
model = SingleConv()
start = time()
for _ in range(N):
    y = model(x)
    loss = y.sum()
    loss.backward()
print(f"PSGs method #1, module hooks: {time() - start}")

##################################
# Method #2: grouped convolution #
##################################

from bastionai.psg.nn import Conv2d

model2 = Conv2d(3, 16, 3, 64)

# Set weight to same value
torch.nn.init.zeros_(model2.weight)
with torch.no_grad():
    model2.weight.add_(model.conv1.weight)

# Set bias to same value
torch.nn.init.zeros_(model2.bias)
with torch.no_grad():
    model2.bias.add_(model.conv1.bias)

start2 = time()
for _ in range(N):
    y2 = model2(x)
    loss2 = y2.sum()
    loss2.backward()
print(f"PSGs method #2, expanded weights: {time() - start2}")
print(f"Output mean delta: {(y - y2).abs().sum() / y.numel()}")
print(f"Output max delta: {(y - y2).abs().max()}")
psg = [g for _, g in model.grad_sample_parameters()][0]
print(F"PSGs #1 size: {psg.size()}")
psg2 = model2.expanded_weight.grad
psg2 = psg2.view(64, 16, 3, 3, 3)
print(F"PSGs #2 size: {psg2.size()}")
print(f"PSGs mean delta: {(psg - psg2).abs().sum() / psg.numel() / N}")
print(f"PSGs max delta: {(psg - psg2).abs().max() / N}")
