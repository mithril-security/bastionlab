#!/usr/bin/env python
# coding: utf-8


import polars as pl
import logging
import unittest
from bastionlab import Connection
from server import launch_server

logging.basicConfig(level=logging.INFO)


class TestingConnection(unittest.TestCase):
    def testingconnection(self):
        connection = Connection("localhost", 50056)
        client = connection.client
        self.assertNotEqual(client, None)


def setUpModule():
    global train_dataset, test_dataset, lreg_model

    launch_server()

    # class LReg(Module):
    #     def __init__(self) -> None:
    #         super().__init__()
    #         self.fc1 = Linear(1, 1, 2)

    #     def forward(self, x: Tensor) -> Tensor:
    #         return self.fc1(x)

    # lreg_model = LReg()

    # X = torch.tensor([[0.0], [1.0], [0.5], [0.2]])
    # Y = torch.tensor([[0.0], [2.0], [1.0], [0.4]])
    # train_dataset = TensorDataset([X], Y)

    # X = torch.tensor([[0.1], [-1.0]])
    # Y = torch.tensor([[0.2], [-2.0]])
    # test_dataset = TensorDataset([X], Y)


if __name__ == "__main__":
    unittest.main()


# df = pl.read_csv("train.csv").limit(50)

# connection = Connection("localhost", 50056)
# client = connection.client

# rdf = client.send_df(df)


# per_class_rates = (
#     rdf.select([pl.col("Pclass"), pl.col("Survived")])
#     .groupby(pl.col("Pclass"))
#     .agg(pl.col("Survived").mean())
#     .sort("Survived", reverse=True)
#     .collect()
#     .fetch()
# )


# per_sex_rates = (
#     rdf.select([pl.col("Sex"), pl.col("Survived")])
#     .groupby(pl.col("Sex"))
#     .agg(pl.col("Survived").mean())
#     .sort("Survived", reverse=True)
#     .collect()
#     .fetch()
# )


# connection.close()
