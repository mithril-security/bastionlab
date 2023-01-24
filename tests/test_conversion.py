from bastionlab.torch.utils import TensorDataset
import polars as pl
import torch
import logging
import unittest
from bastionlab import Connection
from bastionlab.polars.policy import (
    Policy,
    TrueRule,
    Log,
)
import math
from bastionlab.polars import train_test_split

logging.basicConfig(level=logging.INFO)


class TestingConnection(unittest.TestCase):
    def test_connection(self):
        connection = Connection("localhost")
        client = connection.client
        self.assertNotEqual(client, None)
        connection.close()

    def test_df_to_tensor_conv(self):
        connection = Connection("localhost")
        client = connection.client
        df = pl.DataFrame(
            {
                "a": [0, 1, 2, 3, 4],
                "b": [2, 3, 5, 6, 7],
            }
        ).with_column((pl.col("a") * pl.col("b")).alias("c"))

        rdf = client.polars.send_df(df, Policy(TrueRule(), Log(), False))

        arr = rdf.to_array()

        df_tensor = arr.to_tensor()
        torch_tensor = torch.tensor(df.to_numpy())
        self.assertEqual(
            df_tensor.shape, torch_tensor.shape, "Tensors are not the same Shape"
        )
        connection.close()

    def test_split_remote_array(self):
        connection = Connection("localhost")
        client = connection.client
        df = pl.DataFrame(
            {
                "a": [0, 1, 2, 3, 4],
                "b": [2, 3, 5, 6, 7],
            }
        ).with_column((pl.col("a") * pl.col("b")).alias("c"))

        rdf = client.polars.send_df(df, Policy(TrueRule(), Log(), False))

        arr = rdf.to_array()

        ratio = 0.2

        train_X, test_X = train_test_split(arr, test_size=ratio, shuffle=False)

        train_tensor = train_X.to_tensor()
        test_tensor = test_X.to_tensor()

        def splitter(df: pl.DataFrame, ratio: float):
            height = df.height * 1.0
            test_size = int(math.floor(height * ratio))
            train_size = int(math.floor(height * (1 - ratio)))
            return df.head(train_size).to_numpy(), df.tail(test_size).to_numpy()

        sk_train_X, sk_test_X = splitter(df, ratio)

        sk_train_tensor = torch.tensor(sk_train_X)
        sk_test_tensor = torch.tensor(sk_test_X)

        self.assertIsNotNone(train_tensor)
        self.assertIsNotNone(test_tensor)

        self.assertEqual(
            train_tensor.shape, sk_train_tensor.shape, "Train set not the same Shape"
        )
        self.assertEqual(
            test_tensor.shape, sk_test_tensor.shape, "Test set not the same Shape"
        )
        connection.close()

    def test_torch_dataset_upload(self):
        connection = Connection("localhost")
        client = connection.client
        X = torch.tensor([[0.0], [1.0], [0.5], [0.2]])
        Y = torch.tensor([[0.0], [2.0], [1.0], [0.4]])
        train_dataset = TensorDataset([X], Y)

        X = torch.tensor([[0.1], [-1.0]])
        Y = torch.tensor([[0.2], [-2.0]])
        test_dataset = TensorDataset([X], Y)

        train_dataset = client.torch.RemoteDataset(
            train_dataset,
            name="1D Linear Regression",
            description="Dummy 1D Linear Regression Dataset (param is 2)",
            privacy_limit=8320.1,
        )
        test_dataset = client.torch.RemoteDataset(
            test_dataset,
            name="1D Linear Regression",
            description="Dummy 1D Linear Regression Dataset (param is 2)",
            privacy_limit=8320.1,
        )

        train_inputs_shapes = [input.shape for input in train_dataset.inputs]
        test_inputs_shapes = [input.shape for input in test_dataset.inputs]

        self.assertListEqual(
            train_inputs_shapes,
            [
                torch.Size((4, 1)),
            ],
        )
        self.assertListEqual(
            test_inputs_shapes,
            [
                torch.Size((2, 1)),
            ],
        )
        self.assertEqual(
            train_dataset.privacy_limit,
            8320.1,
        )
        self.assertEqual(
            train_dataset.nb_samples,
            4,
        )
        self.assertEqual(
            train_dataset.description, "Dummy 1D Linear Regression Dataset (param is 2)"
        )
        self.assertEqual(
            train_dataset.name,
            "1D Linear Regression",
        )

    def test_send_tensor_method(self):
        connection = Connection("localhost")
        client = connection.client
        X = torch.tensor([[0.0], [1.0], [0.5], [0.2]])
        remote_tensor = client.torch.RemoteTensor(X)

        self.assertEqual(remote_tensor.shape, torch.Size((4, 1)))
        self.assertEqual(remote_tensor.dtype, torch.float)

if __name__ == "__main__":

    unittest.main()
