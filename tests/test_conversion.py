#!/usr/bin/env python
# coding: utf-8


import polars as pl
import torch
from sklearn.model_selection import train_test_split as sk_train_test_split
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


if __name__ == "__main__":

    unittest.main()
