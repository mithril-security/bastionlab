from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple, Union
import polars as pl
from tokenizers import Tokenizer
from ..polars.remote_polars import RemoteSeries, delegate, delegate_properties
from ..torch.remote_torch import RemoteTensor

if TYPE_CHECKING:
    from ..polars.remote_polars import RemoteSeries
    from ..torch.remote_torch import RemoteTensor


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
    _tokenizer: "Tokenizer"

    @staticmethod
    def from_hugging_face_pretrained(model: str) -> "RemoteTokenizer":
        from tokenizers import Tokenizer

        return RemoteTokenizer(Tokenizer.from_pretrained(model))

    def encode(
        self, series: "RemoteSeries", return_tensors: bool = True
    ) -> Tuple[Union[RemoteTensor, RemoteSeries], Union[RemoteTensor, RemoteSeries]]:
        res = series.convert(series.columns, self._tokenizer.to_str()).collect()
        input_ids = res.column(res.columns[0])
        attention_mask = res.column(res.columns[1])

        if return_tensors:
            return input_ids.to_tensor(), attention_mask.to_tensor()
        return input_ids, attention_mask

    def __str__(self) -> str:
        return f"RemoteTokenizer(vocabulary_size={self._tokenizer.get_vocab_size()})"

    def __repr__(self) -> str:
        return str(self)
