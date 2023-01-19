from functools import reduce
import polars as pl
import logging
import unittest
from bastionlab import Connection
from bastionlab.polars.policy import (
    Policy,
    TrueRule,
    Log,
)

logging.basicConfig(level=logging.INFO)


def df_to_list(df):
    return list(df.to_dict().values())[0].to_list()


class TestingConnection(unittest.TestCase):
    def setUp(self) -> None:
        self.connection = Connection("localhost", 50056)
        self.client = self.connection.client
        self.df = pl.DataFrame(
            {
                "a": [
                    "Welcome to ChatGPT. ChatGPT is the best",
                    "ChatGPT is at capacity right now",
                ]
            }
        )
        self.rdf = self.client.polars.send_df(
            self.df,
            Policy(TrueRule(), Log(), False),
        )

    def test_string_ops(self):
        splitter = " "
        remote_splitted = df_to_list(self.rdf.split(splitter).collect().fetch())

        local_splitted = [
            row.split(splitter) for row in self.df.to_numpy().flatten().tolist()
        ]

        self.assertListEqual(remote_splitted, local_splitted)

    def test_match(self):
        remote_matches = df_to_list(self.rdf.match("ChatGPT").collect().fetch())
        local_matches = [True, True]
        self.assertListEqual(remote_matches, local_matches)

    def test_to_lowercase(self):
        remote_lower = df_to_list(self.rdf.to_lowercase().collect().fetch())
        local_lower = [l.lower() for l in df_to_list(self.df)]
        self.assertListEqual(remote_lower, local_lower)

    def test_to_uppercase(self):
        remote_upper = df_to_list(self.rdf.to_uppercase().collect().fetch())
        local_upper = [l.upper() for l in df_to_list(self.df)]
        self.assertListEqual(remote_upper, local_upper)

    def test_replace(self):
        remote_replace = df_to_list(self.rdf.replace("ChatGPT", "AI").collect().fetch())
        local_replace = [l.replace("ChatGPT", "AI", 1) for l in df_to_list(self.df)]
        self.assertListEqual(remote_replace, local_replace)

    def test_replace_all(self):
        remote_replace = df_to_list(
            self.rdf.replace_all("ChatGPT", "AI").collect().fetch()
        )
        local_replace = [l.replace("ChatGPT", "AI") for l in df_to_list(self.df)]
        self.assertListEqual(remote_replace, local_replace)

    def test_contains(self):
        remote_contains = df_to_list(self.rdf.contains("ChatGPT").collect().fetch())
        local_contains = [True, True]
        self.assertListEqual(remote_contains, local_contains)

    def test_findall(self):
        remote_findall = df_to_list(self.rdf.findall("(?i)chatgpt").collect().fetch())
        local_findall = [["ChatGPT", "ChatGPT"], ["ChatGPT"]]

        self.assertListEqual(remote_findall, local_findall)

    def test_extract(self):
        remote_extract = df_to_list(self.rdf.extract("(?i)chatgpt").collect().fetch())
        local_extract = [["ChatGPT", "ChatGPT"], ["ChatGPT"]]

        self.assertListEqual(remote_extract, local_extract)

    def test_fuzzy_good_match(self):
        remote_fuzzy = df_to_list(self.rdf.fuzzy_match("chat").collect().fetch())
        local_match = df_to_list(self.df)
        self.assertListEqual(remote_fuzzy, local_match)

    def test_fuzzy_bad_match(self):
        remote_fuzzy = df_to_list(self.rdf.fuzzy_match("wws").collect().fetch())
        local_match = [None, None]
        self.assertListEqual(remote_fuzzy, local_match)

    def tearDown(self) -> None:
        self.connection.close()


if __name__ == "__main__":
    unittest.main()
