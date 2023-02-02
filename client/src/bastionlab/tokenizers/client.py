from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bastionlab.client import Client
    from .remote_tokenizers import RemoteTokenizer


class BastionLabTokenizers:
    def __init__(self, client: "Client") -> None:
        self.client = client

    def from_hugging_face_pretrained(self, model_name: str) -> "RemoteTokenizer":
        """
        Loads a Hugging Face tokenizer model with the checkpoint name.

        Args:
            model_name: str
                Model name.
        """
        from .remote_tokenizers import RemoteTokenizer

        return RemoteTokenizer._from_hugging_face_pretrained(self.client, model_name)
