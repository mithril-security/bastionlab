from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple, Union
import polars as pl
from tokenizers import Tokenizer
from ..polars.remote_polars import RemoteArray, Metadata, delegate, delegate_properties
from ..torch.remote_torch import RemoteTensor

if TYPE_CHECKING:
    from ..torch.remote_torch import RemoteTensor
    from ..polars.remote_polars import RemoteLazyFrame, RemoteArray, Metadata
    from ..client import Client


@delegate_properties(
    "padding",
    "truncation",
    "model",
    "decoder",
    "pre_tokenizer",
    "post_processor",
    "normalizer",
    target="_tokenizer",
)
@delegate(
    target_cls=Tokenizer,
    target_attr="_tokenizer",
    f_names=[
        "enable_padding",
        "enable_truncation",
        "get_vocab",
        "get_vocab_size",
        "no_padding",
        "add_tokens",
        "post_process",
        "to_str",
    ],
    wrap=False,
)
@dataclass
class RemoteTokenizer:
    _meta: "Metadata"
    _tokenizer: "Tokenizer"

    @staticmethod
    def from_hugging_face_pretrained(client: "Client", model: str) -> "RemoteTokenizer":
        from tokenizers import Tokenizer

        return RemoteTokenizer(Metadata(client), Tokenizer.from_pretrained(model))

    def encode(
        self,
        rdf: "RemoteLazyFrame",
        add_special_tokens=True,
    ) -> Tuple[Union[RemoteTensor, RemoteArray], Union[RemoteTensor, RemoteArray]]:
        res = rdf._convert(
            rdf.column_names, self._tokenizer.to_str(), add_special_tokens
        ).collect()
        input_ids = res.select(res.column_names[0]).to_array()
        attention_mask = res.select(res.column_names[1]).to_array()

        return input_ids, attention_mask

    def __str__(self) -> str:
        return f"RemoteTokenizer(vocabulary_size={self._tokenizer.get_vocab_size()})"

    def __repr__(self) -> str:
        return str(self)
