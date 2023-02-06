from bastionlab.torch.utils import TensorDataset
import polars as pl
import torch
import numpy as np
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


class TestingDataConv(unittest.TestCase):
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
        tensor = arr.to_tensor()
        torch_tensor = torch.tensor(df.to_numpy())
        self.assertEqual(
            tensor.shape, torch_tensor.shape, "Tensors are not the same Shape"
        )
        connection.close()

    def test_split_remote_array_with_negs(self):
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

        with self.assertRaises(ValueError) as ve:
            train_test_split(arr, test_size=-0.4, shuffle=False)

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
            train_size = int(height) - test_size

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

        self.assertEqual(remote_tensor.shape, X.shape)
        self.assertEqual(remote_tensor.dtype, torch.float)

    def test_list_dataframe_same_type(self):
        connection = Connection("localhost")
        client = connection.client
        df = pl.DataFrame(
            {
                "a": [[0, 1, 2, 3, 4]],
                "b": [[2, 3, 5, 6, 7]],
            }
        )

        rdf = client.polars.send_df(df, Policy(TrueRule(), Log(), False))

        remote_tensor = rdf.to_array().to_tensor()
        local_tensor = torch.tensor(df.to_numpy().tolist())

        self.assertEqual(remote_tensor.shape, local_tensor.shape)
        self.assertEqual(remote_tensor.dtype, local_tensor.dtype)

    def test_list_dataframe_of_diff_types(self):
        connection = Connection("localhost")
        client = connection.client
        df = pl.DataFrame(
            {
                "a": [[0, 1, 2, 3, 4]],
                "b": [[2.0, 3.1, 0.5, 1.6, 0.317]],
            }
        )

        rdf = client.polars.send_df(df, Policy(TrueRule(), Log(), False))

        # We expect this test to fail because the dataframe contains lists of different types
        with self.assertRaises(Exception) as context:
            rdf.to_array()

        self.assertTrue(
            "DataTypes for all columns should be the same"
            in context.exception.details()
        )

    def test_dataframe_with_diff_types(self):
        connection = Connection("localhost")
        client = connection.client
        df = pl.DataFrame(
            {
                "a": [0, 1, 2, 3, 4],
                "b": [2.0, 3.1, 0.5, 1.6, 0.317],
            }
        )

        rdf = client.polars.send_df(df, Policy(TrueRule(), Log(), False))

        # We expect this test to fail because the dataframe contains lists of different types
        with self.assertRaises(Exception) as context:
            rdf.to_array()

        self.assertTrue(
            "DataTypes for all columns should be the same"
            in context.exception.details()
        )


if __name__ == "__main__":
    unittest.main()
