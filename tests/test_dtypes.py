#!/usr/bin/env python
# coding: utf-8


import polars as pl
import logging
import unittest
from bastionlab import Connection
from bastionlab.polars.policy import (
    Policy,
    TrueRule,
    Log,
)
from functools import reduce

logging.basicConfig(level=logging.INFO)


def is_truthy(df1, df2):
    res = df1 == df2
    res = res.to_numpy().flatten()
    res = reduce(lambda a, b: a and b, res)
    return res


class TestingConnection(unittest.TestCase):
    def test_connection(self):
        connection = Connection("localhost", 50056)
        client = connection.client
        self.assertNotEqual(client, None)
        connection.close()

    def test_dtypes(self):
        connection = Connection("localhost")
        client = connection.client
        frames = [
            pl.DataFrame({"a": [[1], [2]]}),
            pl.DataFrame({"a": ["b", "a"]}),
            pl.DataFrame({"a": [["b"], ["a"]]}),
            pl.DataFrame({"a": [1, 2, 4]}),
            pl.DataFrame({"a": [None]}),
        ]

        for df in frames:
            rdf = client.polars.send_df(
                df, policy=Policy(TrueRule(), Log(), savable=False, convertable=True)
            )
            df2 = rdf.select(pl.all()).collect().fetch()

            self.assertTrue(is_truthy(df, df2))

    def test_mixed_types_dataframe(self):
        connection = Connection("localhost")
        client = connection.client
        frames = [
            pl.DataFrame({"a": ["dog", "cat"], "b": [["cat", "dog"], ["pig", "goat"]]}),
            pl.DataFrame({"a": [[1]], "b": [1]}),
            pl.DataFrame(
                {
                    "a": ["dog", "cat"],
                    "b": [["cat", "dog"], ["pig", "goat"]],
                    "d": [[1], [2]],
                    "e": [1, 0],
                }
            ),
        ]

        for df in frames:
            rdf = client.polars.send_df(
                df, policy=Policy(TrueRule(), Log(), savable=False, convertable=True)
            )
            df2 = rdf.select(pl.all()).collect().fetch()

            self.assertTrue(is_truthy(df, df2))


if __name__ == "__main__":
    unittest.main()
