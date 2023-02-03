from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from bastionlab.client import Client
    from .remote_tokenizers import RemoteTokenizer


class BastionLabTokenizers:
    def __init__(self, client: "Client") -> None:
        self.client = client

    def from_hugging_face_pretrained(
        self,
        model_name: str,
        revision: Optional[str] = "main",
        auth_token: Optional[str] = None,
    ) -> "RemoteTokenizer":
        """
        Loads a Hugging Face tokenizer model with the checkpoint name.

        Args:
            model_name: str
                Model name.
            revision: str
                A branch or commit id
            auth_token: str, optional, default=None
                An optional auth token used to access private repositories on the Hugging Face Hub
        """
        from .remote_tokenizers import RemoteTokenizer

        return RemoteTokenizer._from_hugging_face_pretrained(
            self.client, model_name, revision, auth_token
        )
