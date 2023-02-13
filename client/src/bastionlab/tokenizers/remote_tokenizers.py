from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple, Union, Optional, Dict
import polars as pl
import json
from tokenizers import Tokenizer
from ..polars.remote_polars import RemoteArray, Metadata, delegate, delegate_properties
from ..pb.bastionlab_conversion_pb2 import ToTokenizedArrays

if TYPE_CHECKING:
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
    target_attr="_tokenizer",
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
    ],
    wrap=False,
)
@dataclass
class RemoteTokenizer:
    _client: "Client"
    _tokenizer: Optional["Tokenizer"] = None
    _model_name: Optional[str] = None
    _revision: Optional[str] = "main"
    _auth_token: Optional[str] = None

    @staticmethod
    def _from_hugging_face_pretrained(
        client: "Client",
        model: str,
        revision: Optional[str] = "main",
        auth_token: Optional[str] = None,
    ) -> "RemoteTokenizer":
        """
        Loads a Hugging Face tokenizer model with the checkpoint name.

        Args:
            model: str
                Model name.
        """
        from tokenizers import Tokenizer

        return RemoteTokenizer(
            client,
            Tokenizer.from_pretrained(model, revision=revision, auth_token=auth_token),
            model,
            revision,
            auth_token,
        )

    def encode(
        self,
        rdf: "RemoteLazyFrame",
        add_special_tokens: bool = True,
    ) -> Tuple[RemoteArray, RemoteArray]:
        """
        Encodes a RemoteLazyFrame as tokenized RemoteArray.

        Args:
            rdf: RemoteLazyFrame
                The RemoteDataframe containing string sequences to be tokenized.
            add_special_tokens: bool
                Whether to add the special tokens

        Returns:
            Tuple[RemoteArray, RemoteArray]
                Returns a tuple of the tokenized entries (first RemoteArray contains input_ids and the other, attention_mask)
        """
        ids, masks = self._client._converter._stub.TokenizeDataFrame(
            ToTokenizedArrays(
                identifier=rdf.identifier,
                add_special_tokens=add_special_tokens,
                model=self._model_name,
                config=self._serialize(),
                revision=self._revision,
                auth_token=self._auth_token,
            )
        ).list

        ids = RemoteArray(self._client, ids.identifier)
        masks = RemoteArray(self._client, masks.identifier)

        return ids, masks

    def __str__(self) -> str:
        return f"RemoteTokenizer(vocabulary_size={self._tokenizer.get_vocab_size()})"

    def __repr__(self) -> str:
        return str(self)

    def _serialize(self) -> Tuple[str, str]:
        padding = _process_padding(self._tokenizer.padding)

        # The logic below is implemented after studying how the `padding` and `truncation` props work
        if self._tokenizer.padding is not None and self._tokenizer.truncation is None:
            self._tokenizer.enable_truncation(
                max_length=self._tokenizer.padding.get("length")
            )
        # For truncation, if it's None, we will forcefully call `enable_truncation`
        truncation = _proccess_truncation(self._tokenizer.truncation)
        return json.dumps(
            dict(
                padding_params=padding,
                truncation_params=truncation,
            )
        )


def from_snake_case_to_camel_case(snake_string: str):
    return snake_string.title().replace("_", "")


def _process_padding(padding: Optional[Dict] = None):
    if padding:
        strategy = padding.get("length")
        strategy = "BatchLongest" if strategy is None else dict(Fixed=strategy)
        direction = from_snake_case_to_camel_case(padding.get("direction"))
        padding.update(direction=direction)
        padding["strategy"] = strategy

    return padding


def _proccess_truncation(truncation: Optional[Dict] = None):
    if truncation:
        truncation.update(
            direction=from_snake_case_to_camel_case(truncation.get("direction"))
        )
        truncation.update(
            strategy=from_snake_case_to_camel_case(truncation.get("strategy"))
        )
    return truncation
