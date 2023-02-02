import polars as pl
import logging
import unittest
import torch
from tokenizers import Tokenizer
from bastionlab import Connection
from bastionlab.polars.policy import (
    Policy,
    TrueRule,
    Log,
)

logging.basicConfig(level=logging.INFO)


class TestingRemoteTokenizer(unittest.TestCase):
    def setUp(self) -> None:
        self.distilbert = "distilbert-base-uncased"

    def test_connection(self):
        connection = Connection("localhost")
        client = connection.client
        self.assertNotEqual(client, None)
        connection.close()

    def test_tokenizers_encode_tensors(self):
        connection = Connection("localhost")
        client = connection.client

        df = pl.DataFrame({"words": ["I am here", "where are you going"]})
        rdf = client.polars.send_df(df, Policy(TrueRule(), Log(), False))

        remote_tokenizer = client.tokenizers.from_hugging_face_pretrained(
            "distilbert-base-uncased"
        )

        local_tokenizer = Tokenizer.from_pretrained("distilbert-base-uncased")

        remote_tokenizer.enable_padding(length=5)
        local_tokenizer.enable_padding(length=5)

        remote_ids, remote_masks = remote_tokenizer.encode(
            rdf, add_special_tokens=False
        )

        remote_ids = remote_ids.to_tensor()
        remote_masks = remote_masks.to_tensor()

        ids_mask = lambda enc: [enc.ids, enc.attention_mask]
        rows = [
            ids_mask(local_tokenizer.encode(sentence, add_special_tokens=False))
            for sentence in df.to_numpy().flatten()
        ]

        local_ids = torch.tensor([row[0] for row in rows])
        local_masks = torch.tensor([row[1] for row in rows])

        self.assertIsNotNone(remote_ids)
        self.assertIsNotNone(remote_masks)

        self.assertEqual(remote_ids.shape, local_ids.shape)
        self.assertEqual(remote_masks.shape, local_masks.shape)
        self.assertEqual(remote_ids.dtype, local_ids.dtype)
        self.assertEqual(remote_masks.dtype, local_masks.dtype)

        connection.close()


if __name__ == "__main__":
    unittest.main()
