import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.conv as conv
from typing import Callable, List, Union, Tuple, Optional, TypeVar
from torch import Tensor, Size


def _single(x: Union[int, Tuple[int]]) -> Tuple[int]:
    if isinstance(x, tuple):
        return x
    return (x,)


def _pair(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(x, tuple):
        return x
    return (x, x)


def _triple(x: Union[int, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    if isinstance(x, tuple):
        return x
    return (x, x, x)


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
    x = module.__getattr__(name).detach()  # type: ignore [operator]
    module.__dict__.get("_parameters").pop(name)  # type: ignore [union-attr]
    module.__dict__.get("_buffers")[name] = value  # type: ignore [index]


def _repeat_int_list(l: List[int], repeats: int) -> List[int]:
    res: List[int] = []
    for _ in range(repeats):
        res += l
    return res


def _pad_input(x: Tensor, max_batch_size: int) -> Tensor:
    batch_size = x.size(0)
    batch_padding = _repeat_int_list([0, 0], len(x.size()) - 1) + [
        0,
        max_batch_size - batch_size,
    ]
    return F.pad(x, batch_padding, "constant", 0.0)


class _ConvNd(conv._ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, ...],
        max_batch_size: int,
        stride: Tuple[int, ...],
        padding: Union[str, Tuple[int, ...]],
        dilation: Tuple[int, ...],
        transposed: bool,
        output_padding: Tuple[int, ...],
        groups: int,
        bias: bool,
        padding_mode: str,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,  # type: ignore [arg-type]
            dilation,
            transposed,
            output_padding,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        self.kernel_dim = len(kernel_size)
        self.max_batch_size = max_batch_size
        _reassign_parameter_as_buffer(self, "weight", self.weight.detach())
        self.expanded_weight = nn.Parameter(
            self.weight.expand(max_batch_size, *self.weight.size())
        )
        if bias:
            _reassign_parameter_as_buffer(self, "bias", self.bias.detach())  # type: ignore [union-attr]
            self.expanded_bias = nn.Parameter(
                self.bias.expand(max_batch_size, *self.bias.size())  # type: ignore [union-attr]
            )
        else:
            del self.bias
            self.bias = None
            self.register_parameter("expanded_bias", None)

    def extra_repr(self):
        s = super().extra_repr()
        s_params = s.split(", ")
        return ", ".join(
            [
                *s_params[: 2 + self.kernel_dim],
                f"max_batch_size={self.max_batch_size}",
                *s_params[2 + self.kernel_dim :],
            ]
        )


T = TypeVar("T")


def expanded_convolution(
    conv_fn: Callable, tuple_fn: Callable[[T], Tuple[int, ...]]
) -> Callable:
    class ConvNd(_ConvNd):
        """Convolutional layer with expanded weights to be used with DP-SGD.

        Weights are expanded to the provided `max_batch_size` so that the autodiff computes
        the per-samples gradient needed by the DP-SGD algorithm.

        Expansion is made without copying or allocating more memory at the model
        lifetime scale as expanded weights are just a view on the original weights
        (similar to broadcasting).

        However, weights are reallocated while computing the forward pass for a short amount of time
        as the forward pass computation needs them in a contiguous format.
        As layers are typically used one after the other, the overall memory impact is neglectable.

        To speed up the computation of the forward pass with expanded weights, we use grouped convolutions
        with a number of groups equal to the number of samples: the convolution operator uses one kernel group per sample
        (which makes sample computations independent) and the weights of these are shared thanks to the expansion.

        Refer to the Pytorch documentation for more on how to use the various parameters:
            1D: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d
            2D: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
            3D: https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html#torch.nn.Conv3d
        """

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: T,
            max_batch_size: int,
            stride: T = 1,  # type: ignore [assignment]
            padding: Union[str, T] = 0,  # type: ignore [assignment]
            dilation: T = 1,  # type: ignore [assignment]
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = "zeros",
            device: Optional[Union[torch.device, str]] = None,
            dtype: Optional[torch.dtype] = None,
        ):
            kernel_size_: Tuple[int, ...] = tuple_fn(kernel_size)
            stride_: Tuple[int, ...] = tuple_fn(stride)
            padding_: Union[str, Tuple[int, ...]] = (
                padding if isinstance(padding, str) else tuple_fn(padding)
            )
            dilation_: Tuple[int, ...] = tuple_fn(dilation)
            super().__init__(
                in_channels,
                out_channels,
                kernel_size_,
                max_batch_size,
                stride_,
                padding_,
                dilation_,
                False,
                tuple_fn(0),  # type: ignore [arg-type]
                groups,
                bias,
                padding_mode,
                device,
                dtype,
            )
            self._zero_padding: Tuple[int, ...] = tuple_fn(0)  # type: ignore [arg-type]

        def _conv_forward(self, x: Tensor, weight: Tensor, bias: Optional[Tensor]):
            batch_size = x.size(0)
            x = _pad_input(x, self.max_batch_size)
            x = x.view(_gconv_size(x))
            weight = weight.reshape(_gconv_size(weight, drop_first=True))
            if bias is not None:
                bias = bias.reshape(_gconv_size(bias, drop_first=True))

            if self.padding_mode != "zeros":
                x = conv_fn(
                    F.pad(
                        x,
                        self._reversed_padding_repeated_twice,
                        mode=self.padding_mode,
                    ),
                    weight,
                    bias,
                    self.stride,
                    self._zero_padding,
                    self.dilation,
                    self.groups * self.max_batch_size,
                )
            x = conv_fn(
                x,
                weight,
                bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups * self.max_batch_size,
            )

            x = x.view(_std_size(x, self.max_batch_size))
            return x[:batch_size]

        def forward(self, x: Tensor) -> Tensor:
            return self._conv_forward(x, self.expanded_weight, self.expanded_bias)

    return ConvNd


Conv1d = expanded_convolution(F.conv1d, _single)
Conv2d = expanded_convolution(F.conv2d, _pair)
Conv3d = expanded_convolution(F.conv3d, _triple)


class ConvLinear(nn.Module):
    """Linear layer with expanded weights that internally uses an expanded 1D convolution.

    Refer to the documentation of convolutions for more about the internals and Pytorch's Linear
    Layer documentation for more about the parameters and their usage.
    """

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
        self.inner = Conv1d(
            in_features,
            out_features,
            1,
            max_batch_size,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), 1)
        x = self.inner(x)
        return x.view(x.size(0), -1)


class Linear(nn.Linear):
    """Linear layer with expanded weights to be used with DP-SGD.

    Weights are expanded to the `max_batch_size` so that the autodif computes
    the per-samples gradient needed by the DP-SGD algorithm.

    Expansion is made without copying or allocating more memory as expanded
    weights are just a view on the original weights (similar to broadcasting).

    However, this implies the forward pass is performed with einsum which may slightly
    decrese the performance of the computation.

    Refer to the Pytorch documentation for more on how to use the various parameters:
    https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_batch_size: int,
        bias: bool = True,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias,
            device,
            dtype,
        )
        self.max_batch_size = max_batch_size
        _reassign_parameter_as_buffer(self, "weight", self.weight.detach())
        self.expanded_weight = nn.Parameter(
            self.weight.expand(max_batch_size, *self.weight.size())
        )
        if bias:
            _reassign_parameter_as_buffer(self, "bias", self.bias.detach())
            self.expanded_bias = nn.Parameter(
                self.bias.expand(max_batch_size, *self.bias.size())
            )
        else:
            del self.bias
            self.bias = None  # type: ignore [assignment]
            self.register_parameter("expanded_bias", None)

    def extra_repr(self):
        s = super().extra_repr()
        s_params = s.split(", ")
        return ", ".join(
            [*s_params[:3], f"max_batch_size={self.max_batch_size}", *s_params[3:]]
        )

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)
        bias_view_size = (
            [self.expanded_bias.size(0)]
            + _repeat_int_list([1], len(x.size()) - 2)
            + [self.expanded_bias.size(1)]
        )
        x = _pad_input(x, self.max_batch_size)
        x = torch.einsum(
            "n...i,nji->n...j", x, self.expanded_weight
        ) + self.expanded_bias.view(bias_view_size)
        return x[:batch_size]


class Embedding(nn.Embedding):
    """Linear layer with expanded weights to be used with DP-SGD.

    Weights are expanded to the `max_batch_size` so that the autodif computes
    the per-samples gradient needed by the DP-SGD algorithm.

    An embedding layer is essentially a lookup table that internally stores all
    the vectors of the vocabulary and returns the vector associated with each input index.
    To compute per-sample gradients, we "copy" the lookup table as many times
    as the maximum number of samples in a batch. The input indexes are offseted
    by their sample number times the vocabulary size before actually looking up
    so that each sample uses a different "copy" of the lookup table.

    The copy of the lookup table is intself costless as we only use an expanded view
    (similar to broadcasting). The runtime cost is low as well as we just need to remap
    the input indexes.

    Refer to the Pytorch documentation for more on how to use the various parameters:
    https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        max_batch_size: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Optional[Tensor] = None,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            _weight,
            device,
            dtype,
        )
        self.max_batch_size = max_batch_size
        _reassign_parameter_as_buffer(self, "weight", self.weight.detach())
        self.expanded_weight = nn.Parameter(
            self.weight.expand(max_batch_size, *self.weight.size())
        )

    def extra_repr(self):
        s = super().extra_repr()
        s_params = s.split(", ")
        return ", ".join(
            [*s_params[:3], f"max_batch_size={self.max_batch_size}", *s_params[3:]]
        )

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)
        batch_offset_view_size = [self.max_batch_size] + _repeat_int_list(
            [1], len(x.size()) - 1
        )
        x = _pad_input(x, self.max_batch_size)
        batch_offset = torch.arange(self.max_batch_size).view(batch_offset_view_size)
        expanded_indexes = batch_offset * self.max_batch_size + x
        embeddings = F.embedding(
            expanded_indexes,
            self.expanded_weight.reshape(
                self.max_batch_size * self.num_embeddings, self.embedding_dim
            ),
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        return embeddings[:batch_size]


class LayerNorm(nn.LayerNorm):
    """LayerNorm layer with expanded weights to be used with DP-SGD.

    Weights are expanded to the `max_batch_size` so that the autodif computes
    the per-samples gradient needed by the DP-SGD algorithm.

    Expansion is made without copying or allocating more memory as expanded
    weights are just a view on the original weights (similar to broadcasting).

    This comes at no additional cost during the forward pass as LayerNorm involves
    an affine elementwise operation that can directly be done with the expanded weights
    with proper views.

    Refer to the Pytorch documentation for more on how to use the various parameters:
    https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm.
    """

    def __init__(
        self,
        normalized_shape: Union[int, List[int], Size],
        max_batch_size: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(
            normalized_shape,
            eps,
            elementwise_affine,
            device,
            dtype,
        )
        self.max_batch_size = max_batch_size
        if self.elementwise_affine:
            _reassign_parameter_as_buffer(self, "weight", self.weight.detach())
            self.expanded_weight = nn.Parameter(
                self.weight.expand(max_batch_size, *self.weight.size())
            )
            _reassign_parameter_as_buffer(self, "bias", self.bias.detach())
            self.expanded_bias = nn.Parameter(
                self.bias.expand(max_batch_size, *self.bias.size())
            )

    def extra_repr(self):
        s = super().extra_repr()
        s_params = s.split(", ")
        return ", ".join(
            [*s_params[:3], f"max_batch_size={self.max_batch_size}", *s_params[3:]]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = F.layer_norm(x, self.normalized_shape, eps=self.eps)
        if self.elementwise_affine:
            batch_size = x.size(0)
            affine_view_size = (
                [self.max_batch_size]
                + _repeat_int_list([1], len(x.size()) - len(self.normalized_shape) - 1)
                + list(self.normalized_shape)
            )
            x = _pad_input(x, self.max_batch_size)
            x = x * self.expanded_weight.view(
                affine_view_size
            ) + self.expanded_bias.view(affine_view_size)
            x = x[:batch_size]
        return x


if __name__ == "__main__":
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
    print(model.expanded_weight.grad.size())

    w = torch.randn(10, 100)

    fc1 = Linear(100, 10, 64)
    torch.nn.init.zeros_(fc1.weight)
    with torch.no_grad():
        fc1.weight.add_(w)

    fc2 = ConvLinear(100, 10, 64)
    torch.nn.init.zeros_(fc2.inner.weight)
    with torch.no_grad():
        fc2.inner.weight.add_(w.unsqueeze(-1))

    print(((fc1(x) - fc2(x)) ** 2).sum() / 1000)
    print(((fc1(x) - x @ w.T) ** 2).sum() / 1000)
    print(((fc2(x) - x @ w.T) ** 2).sum() / 1000)
