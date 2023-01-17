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
    def test_connection(self):
        connection = Connection("localhost", 50056)
        client = connection.client
        self.assertNotEqual(client, None)
        connection.close()

    def test_dtypes(self):
        from bastionlab.polars.policy import Policy, TrueRule, Log
        from functools import reduce

        connection = Connection("localhost")
        client = connection.client
        frames = [
            pl.DataFrame({"a": [[1], [2]]}),
            pl.DataFrame({"a": ["b", "a"]}),
            pl.DataFrame({"a": [["b"], ["a"]]}),
            pl.DataFrame({"a": [1, 2, 4]}),
            pl.DataFrame({"a": [None]}),
        ]

        def is_truthy(df1, df2):
            res = df1 == df2
            res = res.to_numpy().flatten()
            res = reduce(lambda a, b: a and b, res)
            return res

        for df in frames:
            rdf = client.polars.send_df(
                df, policy=Policy(TrueRule(), Log(), savable=False)
            )
            df2 = rdf.select(pl.all()).collect().fetch()

            self.assertTrue(is_truthy(df, df2))


if __name__ == "__main__":
    unittest.main()
