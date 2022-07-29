import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.conv as conv
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _single, _pair, _triple
from typing import Callable, List, Type, Union, Tuple, Optional, TypeVar
from torch import Tensor, tensor

def _gconv_size(tensor: Tensor, drop_first: bool = False) -> List[int]:
    s = list(tensor.size())
    s[0] = 1
    s[1] = -1
    return s[1:] if drop_first else s

def _std_size(tensor: Tensor, max_batch_size: int) -> List[int]:
    s = list(tensor.size())
    s[0] = max_batch_size
    s[1] = -1
    return s

def _reassign_parameter_as_buffer(module: nn.Module, name: str, value: Tensor) -> None:
    x = module.__getattr__(name).detach()
    module.__dict__.get('_parameters').pop(name)
    module.__dict__.get('_buffers')[name] = value

class _ConvNd(conv._ConvNd):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, ...],
        max_batch_size: int,
        stride: Tuple[int, ...],
        padding: Tuple[int, ...],
        dilation: Tuple[int, ...],
        transposed: bool,
        output_padding: Tuple[int, ...],
        groups: int,
        bias: bool,
        padding_mode: str,
        device=None,
        dtype=None
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            bias,
            padding_mode,
            device,
            dtype
        )
        self.max_batch_size = max_batch_size
        _reassign_parameter_as_buffer(self, 'weight', self.weight.detach())
        self.expanded_weight = nn.Parameter(self.weight.expand(max_batch_size, *self.weight.size()))
        if bias:
            _reassign_parameter_as_buffer(self, 'bias', self.bias.detach())
            self.expanded_bias = nn.Parameter(self.bias.expand(max_batch_size, *self.bias.size()))
        else:
            del self.bias
            self.bias = None
            self.register_parameter('expanded_bias', None)

    def extra_repr(self):
        s = super().extra_repr()
        s_params = s.split(',')
        return ", ".join([*s_params[:3], f"max_batch_size={self.max_batch_size}", *s_params[3:]])

def expanded_convolution(conv_fn: Callable, tuple_type: Type, tuple_fn: Callable) -> Callable:
    def inner(cls: Callable) -> Callable:
        T = TypeVar('T', bound=tuple_type)

        def init(
            _self,
            in_channels: int,
            out_channels: int,
            kernel_size: T,
            max_batch_size: int,
            stride: T = 1,
            padding: Union[str, T] = 0,
            dilation: T = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            device: Optional[Union[torch.device, str]] = None,
            dtype: Optional[torch.dtype] = None,
        ):
            kernel_size_ = tuple_fn(kernel_size)
            stride_ = tuple_fn(stride)
            padding_ = padding if isinstance(padding, str) else tuple_fn(padding)
            dilation_ = tuple_fn(dilation)
            super(cls, _self).__init__(
                in_channels,
                out_channels,
                kernel_size_,
                max_batch_size,
                stride_,
                padding_,
                dilation_,
                False,
                tuple_fn(0),
                groups,
                bias,
                padding_mode,
                device,
                dtype,
            )
        
        def _conv_forward(_self, x: Tensor, weight: Tensor, bias: Optional[Tensor]):
            batch_size = x.size(0)
            batch_padding = [0, 0] * (len(_self.kernel_size) + 1) + [0, _self.max_batch_size - batch_size]
            x = F.pad(x, batch_padding, "constant", 0.)
            x = x.view(_gconv_size(x))
            weight = weight.reshape(_gconv_size(weight, drop_first=True))
            if bias is not None:
                bias = bias.reshape(_gconv_size(bias, drop_first=True))
            
            if _self.padding_mode != 'zeros':
                x = conv_fn(F.pad(x, _self._reversed_padding_repeated_twice, mode=_self.padding_mode),
                                weight, bias, _self.stride,
                                tuple_fn(0), _self.dilation, _self.groups * _self.max_batch_size)
            x = conv_fn(x, weight, bias, _self.stride,
                            _self.padding, _self.dilation, _self.groups * _self.max_batch_size)
            
            x = x.view(_std_size(x, _self.max_batch_size))
            return x[:batch_size]

        def forward(_self, x: Tensor) -> Tensor:
            return _self._conv_forward(x, _self.expanded_weight, _self.expanded_bias)
        
        cls.__init__ = init
        cls._conv_forward = _conv_forward
        cls.forward = forward

        return cls
    return inner
            

@expanded_convolution(F.conv1d, _size_1_t, _single)
class Conv1d(_ConvNd):
    pass

@expanded_convolution(F.conv2d, _size_2_t, _pair)
class Conv2d(_ConvNd):
    pass

@expanded_convolution(F.conv3d, _size_3_t, _triple)
class Conv3d(_ConvNd):
    pass


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
    model = torch.jit.script(Conv2d(3, 16, 3, 64))
    x = torch.randn(64, 3, 256, 256)
    y = model(x)
    loss = y.sum()
    loss.backward()
    print(model.expanded_weight.grad.size())
    print([x.size() for x in model.parameters()])

    model = Linear(100, 10, 64)
    x = torch.randn(64, 100)
    y = model(x)
    loss = y.sum()
    loss.backward()
    print(model.inner.expanded_weight.grad.size())
