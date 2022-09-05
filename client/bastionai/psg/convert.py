import torch
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
        torch.nn.init.zeros_(destination_layer.weight)
        with torch.no_grad():
            destination_layer.weight.add_(source_layer.weight)

    if (
        hasattr(source_layer, "bias")
        and source_layer.bias is not None
        and hasattr(destination_layer, "bias")
        and destination_layer.bias is not None
    ):
        # Set bias to pretrained value
        torch.nn.init.zeros_(destination_layer.bias)
        with torch.no_grad():
            destination_layer.bias.add_(source_layer.bias)


def expand_weights(module: torch.nn.Module, max_batch_size: int) -> None:
    """
    Recursively converts the layers of a model to their expanded counterpart in `bastionai.psg.nn`.

    Args:
        module: model whose weights must be expanded.
        max_batch_size: maximum size of the batches that will be processed by the model.
    """
    # go through all attributes of module nn.module (e.g. network or layer) and put batch norms if present
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.Linear:
            linear_layer = Linear(
                in_features=target_attr.in_features,
                out_features=target_attr.out_features,
                max_batch_size=max_batch_size,
                bias=target_attr.bias is not None,
            )
            _set_weight_and_bias(linear_layer, target_attr)
            setattr(module, attr_str, linear_layer)
        elif type(target_attr) == torch.nn.Conv1d:
            conv_layer = Conv1d(
                in_channels=target_attr.in_channels,
                out_channels=target_attr.out_channels,
                kernel_size=target_attr.kernel_size,
                max_batch_size=max_batch_size,
                stride=target_attr.stride,
                padding=target_attr.padding,
                dilation=target_attr.dilation,
                groups=target_attr.groups,
                bias=target_attr.bias is not None,
                padding_mode=target_attr.padding_mode,
            )
            _set_weight_and_bias(conv_layer, target_attr)
            setattr(module, attr_str, conv_layer)
        elif type(target_attr) == torch.nn.Conv2d:
            conv_layer = Conv2d(
                in_channels=target_attr.in_channels,
                out_channels=target_attr.out_channels,
                kernel_size=target_attr.kernel_size,
                max_batch_size=max_batch_size,
                stride=target_attr.stride,
                padding=target_attr.padding,
                dilation=target_attr.dilation,
                groups=target_attr.groups,
                bias=target_attr.bias is not None,
                padding_mode=target_attr.padding_mode,
            )
            _set_weight_and_bias(conv_layer, target_attr)
            setattr(module, attr_str, conv_layer)
        elif type(target_attr) == torch.nn.Conv3d:
            conv_layer = Conv3d(
                in_channels=target_attr.in_channels,
                out_channels=target_attr.out_channels,
                kernel_size=target_attr.kernel_size,
                max_batch_size=max_batch_size,
                stride=target_attr.stride,
                padding=target_attr.padding,
                dilation=target_attr.dilation,
                groups=target_attr.groups,
                bias=target_attr.bias is not None,
                padding_mode=target_attr.padding_mode,
            )
            _set_weight_and_bias(conv_layer, target_attr)
            setattr(module, attr_str, conv_layer)
        elif type(target_attr) == torch.nn.Embedding:
            embedding_layer = Embedding(
                num_embeddings=target_attr.num_embeddings,
                embedding_dim=target_attr.embedding_dim,
                max_batch_size=max_batch_size,
                padding_idx=target_attr.padding_idx,
                max_norm=target_attr.max_norm,
                norm_type=target_attr.norm_type,
                scale_grad_by_freq=target_attr.scale_grad_by_freq,
                sparse=target_attr.sparse,
            )
            _set_weight_and_bias(embedding_layer, target_attr)
            setattr(module, attr_str, embedding_layer)
        elif type(target_attr) == torch.nn.LayerNorm:
            norm_layer = LayerNorm(
                normalized_shape=list(target_attr.normalized_shape),
                max_batch_size=max_batch_size,
                eps=target_attr.eps,
                elementwise_affine=target_attr.elementwise_affine,
            )

            if target_attr.elementwise_affine:
                _set_weight_and_bias(norm_layer, target_attr)
            setattr(module, attr_str, norm_layer)
        elif type(target_attr) in [
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
        ]:
            setattr(module, attr_str, torch.nn.Identity())

    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    for _, immediate_child_module in module.named_children():
        expand_weights(immediate_child_module, max_batch_size)
