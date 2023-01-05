#!/usr/bin/env python
# coding: utf-8


import polars as pl
import logging
import unittest
from bastionlab import Connection
from bastionlab.polars.policy import (
    Policy,
    Aggregation,
    Log,
)

# from server import launch_server

logging.basicConfig(level=logging.INFO)


class TestingConnection(unittest.TestCase):
    def testingconnection(self):
        connection = Connection("localhost", 50056)
        client = connection.client
        self.assertNotEqual(client, None)
        connection.close()

    def testingdf(self):
        df = pl.read_csv("titanic.csv").limit(50)
        connection = Connection("localhost", 50056)
        client = connection.client
        policy = Policy(
            safe_zone=Aggregation(min_agg_size=1), unsafe_handling=Log(), savable=False
        )
        rdf = client.polars.send_df(df, policy)
        self.assertNotEqual(rdf, None)
        connection.close()

    def testingquery(self):
        df = pl.read_csv("titanic.csv").limit(50)
        connection = Connection("localhost", 50056)
        client = connection.client
        policy = Policy(
            safe_zone=Aggregation(min_agg_size=1), unsafe_handling=Log(), savable=False
        )
        rdf = client.polars.send_df(df, policy)
        per_class_rates = (
            rdf.select([pl.col("Pclass"), pl.col("Survived")])
            .groupby(pl.col("Pclass"))
            .agg(pl.col("Survived").mean())
            .sort("Survived", reverse=True)
            .collect()
            .fetch()
        )
        self.assertNotEqual(per_class_rates.is_empty(), True)

    def testingquery2(self):
        df = pl.read_csv("titanic.csv").limit(50)
        connection = Connection("localhost", 50056)
        client = connection.client
        policy = Policy(safe_zone=Aggregation(1), unsafe_handling=Log(), savable=False)
        rdf = client.polars.send_df(df, policy)
        per_sex_rates = (
            rdf.select([pl.col("Sex"), pl.col("Survived")])
            .groupby(pl.col("Sex"))
            .agg(pl.col("Survived").mean())
            .sort("Survived", reverse=True)
            .collect()
            .fetch()
        )
        self.assertNotEqual(per_sex_rates.is_empty(), True)


def setUpModule():
    print("Hello world")


if __name__ == "__main__":
    unittest.main()
