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

# Method #1: Hooks

engine = PrivacyEngine()

@engine.grad_sample_fn(Conv2d)
def conv2d_grad_sample_fn(layer: Module, input: Tuple[Tensor], grad_output: Tuple[Tensor]) -> Dict[Parameter, Tensor]:
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
        raise(Exception("Unimplemented"))
    # elif type(layer) == nn.Conv1d:
    #     activations = activations.unsqueeze(-2)  # add the H dimension
    #     # set arguments to tuples with appropriate second element
    #     activations = torch.nn.functional.unfold(
    #         activations,
    #         kernel_size=(1, layer.kernel_size[0]),
    #         padding=(0, layer.padding[0]),
    #         stride=(1, layer.stride[0]),
    #         dilation=(1, layer.dilation[0]),
    #     )
    # elif type(layer) == nn.Conv3d:
    #     activations = unfold3d(
    #         activations,
    #         kernel_size=layer.kernel_size,
    #         padding=layer.padding,
    #         stride=layer.stride,
    #         dilation=layer.dilation,
    #     )
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
weight = torch.clone(model.conv1.weight)
bias = torch.clone(model.conv1.bias)
start = time()
for _ in range(N):
    y = model(x)
    loss = y.sum()
    loss.backward()
print(f"PSGs method #1, module hooks: {time() - start}")

# print(model.grad_sample_store[model.conv1.weight].size())

# Method #2: grouped convolution

# W = torch.randn(16, 3, 3, 3)
# B = torch.randn(16)

# class NormalConv(Module):
#     def __init__(self, in_channels, out_channels, kernel_size):
#         super().__init__()
#         # self.weight = Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size))
#         # self.bias = Parameter(torch.zeros(out_channels))
#         self.weight = Parameter(W)
#         self.bias = Parameter(B)

#     def forward(self, x):
#         return conv2d(x, self.weight, self.bias)

class GroupedConv(Module):
    def __init__(self, in_channels, out_channels, kernel_size, batch_size):
        super().__init__()
        # self.weight = Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size))
        # self.bias = Parameter(torch.zeros(out_channels))
        # self.weight = Parameter(W)
        # self.bias = Parameter(B)
        self.weight = weight
        self.bias = bias
        # we have to copy memory here because first dim is not 1 (i.e. cannot be broadcasted) -> maybe use 3d conv instead?
        self.expanded_weight = Parameter(self.weight.repeat(batch_size, 1, 1, 1))
        self.expanded_bias = Parameter(self.bias.repeat(batch_size))
    
    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(1, b * c, h, w)
        x = conv2d(x, self.expanded_weight, self.expanded_bias, groups=b)
        return x.view(b, -1, *x.size()[2:])

# model = NormalConv(3, 16, 3)
# y = model(x)
# print(y[0,0,:,:])

model2 = GroupedConv(3, 16, 3, 64)
# from psg.nn import Conv2dXXX
# model2 = Conv2dXXX(3, 16, 3, 64)
# y2 = model2(x)
# print(y2[0,0,:,:])
# print((y[0,0,:,:] - y2[0,0,:,:]).abs().sum() / 256 / 256)
# print((y - y2).abs().sum() / (16 * 64 * 256 * 256))
# print(((y - y2).abs().clamp(min=0.001) - 0.001).sum())

# model = GroupedConv(3, 16, 3, 64)
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
psg2 = psg2.view(64, -1, *psg2.size()[1:])
print(F"PSGs #2 size: {psg2.size()}")
print(f"PSGs mean delta: {(psg - psg2).abs().sum() / psg.numel() / N}")
print(f"PSGs max delta: {(psg - psg2).abs().max() / N}")
