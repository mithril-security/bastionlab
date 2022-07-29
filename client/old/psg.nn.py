import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Type, Union, Tuple, Optional
from torch import Tensor

def expanded_convolution(layer_class: Callable) -> Callable:
    class Inner(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, ...]],
            max_batch_size: int,
            stride: Union[int, Tuple[int, ...]] = 1,
            padding: Union[str, int, Tuple[int, ...]] = 0,
            dilation: Union[int, Tuple[int, ...]] = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            device: Optional[Union[torch.device, str]] = None,
            dtype: Optional[torch.dtype] = None,
        ):
            super().__init__()
            self.inner = layer_class(
                in_channels = max_batch_size * in_channels,
                out_channels = max_batch_size * out_channels,
                kernel_size = kernel_size,
                stride = stride,
                padding = padding,
                dilation = dilation,
                groups = max_batch_size * groups,
                bias = bias,
                padding_mode = padding_mode,
                device = device,
                dtype = dtype,
            )
            self.max_batch_size = max_batch_size
        
        def forward(self, x):
            # Note: we use some typing tricks to circumvent
            # the limitations of the torch script compiler
            batch_size = x.size(0)
            channels = x.size(1)

            padding_size = list(x.size())
            padding_size[0] = self.max_batch_size - batch_size
            padding = torch.zeros(padding_size)
            x = torch.cat((x, padding))
            
            expanded_size = list(x.size())
            expanded_size[0] = 1
            expanded_size[1] = self.max_batch_size * channels
            x = x.view(expanded_size)
            
            x = self.inner(x)

            output_size = list(x.size())
            output_size[0] = self.max_batch_size
            output_size[1] = -1
            x = x.view(output_size)
            return x[:batch_size]
        
    return Inner

Conv1d = expanded_convolution(nn.Conv1d)
Conv2d = expanded_convolution(nn.Conv2d)
Conv3d = expanded_convolution(nn.Conv3d)

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_batch_size: int,
        bias: bool = True,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.inner = Conv1d(in_features, out_features, 1, max_batch_size, bias=bias, device=device, dtype=dtype)
    
    def forward(self, x):
        x = x.view(x.size(0), x.size(1), 1)
        x = self.inner(x)
        return x.view(x.size(0), -1)

if __name__ == '__main__':
    model = Conv2d(3, 16, 3, 64)
    # model = torch.jit.script(Conv2d(3, 16, 3, 64))
    x = torch.randn(64, 3, 256, 256)
    y = model(x)
    loss = y.sum()
    loss.backward()
    print(model.inner.weight.grad.size())

    model = Linear(100, 10, 64)
    x = torch.randn(64, 100)
    y = model(x)
    loss = y.sum()
    loss.backward()
    print(model.inner.inner.weight.grad.size())
