import torch
from typing import Tuple, Optional
from .nn import LayerNorm, Linear, Embedding, Conv1d, Conv2d, Conv3d


def _set_weight_and_bias(
    destination_layer: torch.nn.Module, source_layer: torch.nn.Module
) -> None:

    if (
        hasattr(source_layer, "weight")
        and source_layer.weight is not None
        and hasattr(destination_layer, "weight")
        and destination_layer.weight is not None
    ):
        # Set weight to pretrained value
        torch.nn.init.zeros_(destination_layer.weight)  # type: ignore [arg-type]
        with torch.no_grad():
            destination_layer.weight.add_(source_layer.weight)  # type: ignore [operator, arg-type]

    if (
        hasattr(source_layer, "bias")
        and source_layer.bias is not None
        and hasattr(destination_layer, "bias")
        and destination_layer.bias is not None
    ):
        # Set bias to pretrained value
        torch.nn.init.zeros_(destination_layer.bias)  # type: ignore [arg-type]
        with torch.no_grad():
            destination_layer.bias.add_(source_layer.bias)  # type: ignore [operator, arg-type]


def parent_layer(module, name: str) -> Tuple[torch.nn.Module, str]:
    """Returns the parent module of the layer corresponding to the given name (path)
    in the given module and the attribute name of the layer wrt its parent.
    """
    segments = name.strip().split(".")
    child_name = segments[-1]

    layer = module
    for segment in segments[:-1]:
        if not segment.isnumeric():
            layer = getattr(layer, segment)
        else:
            layer = layer[int(segment)]

    return (layer, child_name)


def expand_layer(
    layer: torch.nn.Module, max_batch_size: int
) -> Optional[torch.nn.Module]:
    """Returns an expanded version of given layer if supported, else None."""
    if isinstance(layer, torch.nn.Linear):
        return Linear(
            in_features=layer.in_features,
            out_features=layer.out_features,
            max_batch_size=max_batch_size,
            bias=layer.bias is not None,
        )
    elif isinstance(layer, torch.nn.Conv1d):
        return Conv1d(
            in_channels=layer.in_channels,
            out_channels=layer.out_channels,
            kernel_size=layer.kernel_size,
            max_batch_size=max_batch_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
            bias=layer.bias is not None,
            padding_mode=layer.padding_mode,
        )
    elif isinstance(layer, torch.nn.Conv2d):
        return Conv2d(
            in_channels=layer.in_channels,
            out_channels=layer.out_channels,
            kernel_size=layer.kernel_size,
            max_batch_size=max_batch_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
            bias=layer.bias is not None,
            padding_mode=layer.padding_mode,
        )
    elif isinstance(layer, torch.nn.Conv3d):
        return Conv3d(
            in_channels=layer.in_channels,
            out_channels=layer.out_channels,
            kernel_size=layer.kernel_size,
            max_batch_size=max_batch_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
            bias=layer.bias is not None,
            padding_mode=layer.padding_mode,
        )
    elif isinstance(layer, torch.nn.Embedding):
        return Embedding(
            num_embeddings=layer.num_embeddings,
            embedding_dim=layer.embedding_dim,
            max_batch_size=max_batch_size,
            padding_idx=layer.padding_idx,
            max_norm=layer.max_norm,
            norm_type=layer.norm_type,
            scale_grad_by_freq=layer.scale_grad_by_freq,
            sparse=layer.sparse,
        )
    elif isinstance(layer, torch.nn.LayerNorm):
        return LayerNorm(
            normalized_shape=list(layer.normalized_shape),
            max_batch_size=max_batch_size,
            eps=layer.eps,
            elementwise_affine=layer.elementwise_affine,
        )
    elif any(
        [
            isinstance(layer, t)
            for t in [
                torch.nn.BatchNorm1d,
                torch.nn.BatchNorm2d,
                torch.nn.BatchNorm3d,
            ]
        ]
    ):
        return torch.nn.Identity()
    else:
        return None


def expand_weights(module: torch.nn.Module, max_batch_size: int) -> None:
    """Recursively converts the layers of a model to their expanded counterpart in `bastionlab.torch.psg.nn`.

    Args:
        module: model whose weights must be expanded.
        max_batch_size: maximum size of the batches that will be processed by the model.
    """
    for name, layer in module.named_modules():
        expanded_layer = expand_layer(layer, max_batch_size)
        if expanded_layer is not None:
            _set_weight_and_bias(expanded_layer, layer)
            setattr(*parent_layer(module, name), expanded_layer)
