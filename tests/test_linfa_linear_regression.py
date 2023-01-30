import os
import subprocess
import polars as pl
import logging
import numpy as np
import unittest
from bastionlab import Connection
from bastionlab.polars.policy import (
    Policy,
    TrueRule,
    Log,
)
from bastionlab.linfa.trainers import LinearRegression

logging.basicConfig(level=logging.INFO)

TEST_SIZE = 0.20
SHUFFLE = False


def get_datasets():
    cmds = [
        'wget -O diabetes_train.csv.gz "https://github.com/scikit-learn/scikit-learn/raw/98cf537f5c538fdbc9d27b851cf03ce7611b8a48/sklearn/datasets/data/diabetes_data_raw.csv.gz"',
        'wget -O diabetes_target.csv.gz "https://github.com/scikit-learn/scikit-learn/raw/98cf537f5c538fdbc9d27b851cf03ce7611b8a48/sklearn/datasets/data/diabetes_target.csv.gz"',
        "gzip -d diabetes_train.csv.gz",
        "gzip -d diabetes_target.csv.gz",
    ]

    def runcmd(cmd, verbose=False, *args, **kwargs):

        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True
        )
        std_out, std_err = process.communicate()
        if verbose:
            print(std_out.strip(), std_err)
        pass

    if not os.path.exists("diabetes_train.csv"):
        print("Downloading datasets")
        for cmd in cmds:
            runcmd(cmd)


def sklearn_lin_reg(train: np.array, target: np.array):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    train_X, test_X, train_Y, test_Y = train_test_split(
        train,
        target,
        shuffle=SHUFFLE,
        test_size=TEST_SIZE,
        random_state=0,
    )

    lr = LinearRegression()

    lr.fit(train_X, train_Y)
    return lr.predict(test_X), test_Y


class TestingLinfaAlgos(unittest.TestCase):
    def setUp(self) -> None:
        get_datasets()
        self.train_df = pl.read_csv(
            "diabetes_train.csv",
            sep=" ",
            has_header=False,
            new_columns=["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"],
        )
        self.target_df = pl.read_csv(
            "diabetes_target.csv",
            has_header=False,
            new_columns=["target"],
        )
        self.connection = Connection("localhost")
        self.client = self.connection.client
        policy = Policy(safe_zone=TrueRule(), unsafe_handling=Log(), savable=False)
        self.train_rdf = self.client.polars.send_df(self.train_df, policy)
        self.target_rdf = self.client.polars.send_df(self.target_df, policy)

    def test_linreg(self):
        from bastionlab.polars import train_test_split

        lr = LinearRegression()

        self.assertIsNotNone(lr)

        train_X, test_X, train_Y, test_Y = train_test_split(
            self.train_rdf,
            self.target_rdf,
            shuffle=SHUFFLE,
            test_size=TEST_SIZE,
            random_state=0,
        )

        self.assertListEqual(
            [False, False, False, False],
            [train_X is None, test_X is None, train_Y is None, test_Y is None],
        )

        lr.fit(train_X, train_Y)

        linfa_pred_Y = lr.predict(test_X)
        sklearn_pred_Y, sklearn_test_Y = sklearn_lin_reg(
            self.train_df.to_numpy(), self.target_df.to_numpy()
        )
        linfa_pred = linfa_pred_Y.collect().fetch().to_numpy()
        print(linfa_pred, sklearn_pred_Y)
        diff = linfa_pred - sklearn_pred_Y
        abs_diff = np.abs(diff)
        print(abs_diff.max())
        # self.assertEqual(diff_with_ref, 1.0231815394945443e-12)

        # self.assertListEqual(
        #     sklearn_test_Y.flatten().tolist(),
        #     test_Y.collect().fetch().to_numpy().flatten().tolist(),
        # )

    # def

    def tearDown(self) -> None:
        self.connection.close()
        return super().tearDown()


if __name__ == "__main__":
    unittest.main()
