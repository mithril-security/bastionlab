import torch
from .nn import LayerNorm, Linear, Embedding


def expand_weights(module, max_batch_size):
    """
    Recursively put desired batch norm in nn.module module.

    set module = net to start code.
    """
    # go through all attributes of module nn.module (e.g. network or layer) and put batch norms if present
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.Linear:
            in_features = target_attr.in_features
            out_features = target_attr.out_features
            bias = target_attr.bias is not None

            new_layer = Linear(in_features, out_features, max_batch_size, bias)

            # Set weight to pretrained value
            torch.nn.init.zeros_(new_layer.weight)
            with torch.no_grad():
                new_layer.weight.add_(target_attr.weight)

            # Set bias to pretrained value
            torch.nn.init.zeros_(new_layer.bias)
            with torch.no_grad():
                new_layer.bias.add_(target_attr.bias)

            setattr(module, attr_str, new_layer)
        elif type(target_attr) == torch.nn.Embedding:
            num_embeddings = target_attr.num_embeddings
            embedding_dim = target_attr.embedding_dim
            padding_idx = target_attr.padding_idx
            max_norm = target_attr.max_norm
            norm_type = target_attr.norm_type
            scale_grad_by_freq = target_attr.scale_grad_by_freq
            sparse = target_attr.sparse

            new_layer = Embedding(
                num_embeddings,
                embedding_dim,
                max_batch_size,
                padding_idx,
                max_norm,
                norm_type,
                scale_grad_by_freq,
                sparse,
            )

            # Set weight to pretrained value
            torch.nn.init.zeros_(new_layer.weight)
            with torch.no_grad():
                new_layer.weight.add_(target_attr.weight)

            setattr(module, attr_str, new_layer)
        elif type(target_attr) == torch.nn.LayerNorm:
            normalized_shape = target_attr.normalized_shape
            eps = target_attr.eps
            elementwise_affine = target_attr.elementwise_affine

            new_layer = LayerNorm(
                normalized_shape,
                max_batch_size,
                eps,
                elementwise_affine,
            )

            if elementwise_affine:
                # Set weight to pretrained value
                torch.nn.init.zeros_(new_layer.weight)
                with torch.no_grad():
                    new_layer.weight.add_(target_attr.weight)

                # Set bias to pretrained value
                torch.nn.init.zeros_(new_layer.bias)
                with torch.no_grad():
                    new_layer.bias.add_(target_attr.bias)

            setattr(module, attr_str, new_layer)

    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    for _, immediate_child_module in module.named_children():
        expand_weights(immediate_child_module, max_batch_size)
