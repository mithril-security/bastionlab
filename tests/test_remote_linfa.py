import unittest
import subprocess
import os
import polars as pl
from bastionlab import Connection
from bastionlab.polars.policy import TrueRule, Log, Policy
from bastionlab.linfa import LinearRegression, LogisticRegression
from bastionlab.polars import train_test_split
from sklearn.model_selection import train_test_split as sk_train_test_split
import numpy as np


def runcmd(cmd, verbose=False):
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass


def get_datasets():
    # Diabetes dataset
    diabetes_train_link = "https://github.com/scikit-learn/scikit-learn/raw/98cf537f5c538fdbc9d27b851cf03ce7611b8a48/sklearn/datasets/data/diabetes_data_raw.csv.gz"
    diabetes_targets_link = "https://github.com/scikit-learn/scikit-learn/raw/98cf537f5c538fdbc9d27b851cf03ce7611b8a48/sklearn/datasets/data/diabetes_target.csv.gz"

    # Iris Dataset
    iris_dataset = "iris.csv"
    iris_dataset_link = "https://raw.githubusercontent.com/scikit-learn/scikit-learn/98cf537f5c538fdbc9d27b851cf03ce7611b8a48/sklearn/datasets/data/iris.csv"

    diabetes_train = "diabetes_train.csv"
    diabetes_targets = "diabetes_targets.csv"
    cmds = [
        f"wget -O {diabetes_train}.gz {diabetes_train_link}",
        f"wget -O {diabetes_targets}.gz {diabetes_targets_link}",
        f"wget -O {iris_dataset} {iris_dataset_link}",
        f"gzip -d {diabetes_train}.gz",
        f"gzip -d {diabetes_targets}.gz",
    ]

    for cmd in cmds:
        cmd_parts = cmd.split(" ")
        filename = cmd_parts[2].split(".gz")[0]
        if not os.path.exists(filename):
            if cmd_parts[0] == "wget":
                print(f"Downloading Dataset: {filename}")
            runcmd(cmd)


class TestingRemoteLinfa(unittest.TestCase):
    setup_once = False

    def setUp(self) -> None:
        get_datasets()

        self.diabetes_train = pl.read_csv(
            "diabetes_train.csv",
            sep=" ",
            has_header=False,
            new_columns=["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"],
        )
        self.diabetes_targets = pl.read_csv(
            "diabetes_targets.csv",
            has_header=False,
            new_columns=["target"],
        )

        self.iris = pl.read_csv("iris.csv")

        # We will first shuffle DataFrame because the targets were arranged in terms of classes
        self.iris = self.iris.to_numpy()
        np.random.shuffle(self.iris)
        self.iris = pl.DataFrame(self.iris)

        self.connection = Connection("localhost")

        policy = Policy(TrueRule(), Log(), False)

        self.client = self.connection.client
        self.diabetes_train_rdf = self.client.polars.send_df(
            self.diabetes_train, policy
        )

        self.diabetes_train_array = (
            self.diabetes_train_rdf.select(pl.all().cast(pl.Float64))
            .collect()
            .to_array()
        )

        self.diabetes_targets_rdf = self.client.polars.send_df(
            self.diabetes_targets, policy
        ).collect()

        iris_cols = self.iris.columns
        self.iris_train_array = self.client.polars.send_df(
            self.iris.select(iris_cols[:-1]), policy
        ).to_array()

        self.iris_targets_array = self.client.polars.send_df(
            self.iris.select(pl.col(iris_cols[-1]).cast(pl.UInt64)), policy
        ).to_array()

        self.diabetes_targets_array = self.diabetes_targets_rdf.to_array()

        (
            self.diabetes_train_X,
            self.diabetes_test_X,
            self.diabetes_train_Y,
            self.diabetes_test_Y,
        ) = train_test_split(
            self.diabetes_train_array,
            self.diabetes_targets_array,
            test_size=0.2,
            shuffle=False,
        )

        (
            self.diabetes_local_train_X,
            self.diabetes_local_test_X,
            self.diabetes_local_train_Y,
            self.diabetes_local_test_Y,
        ) = sk_train_test_split(
            self.diabetes_train.to_numpy(),
            self.diabetes_targets.to_numpy(),
            test_size=0.2,
            shuffle=False,
        )

        (
            self.iris_train_X,
            self.iris_test_X,
            self.iris_train_Y,
            self.iris_test_Y,
        ) = train_test_split(
            self.iris_train_array, self.iris_targets_array, test_size=0.2, shuffle=False
        )

        print(self.iris.select(iris_cols[-1]).to_numpy().shape)
        (
            self.local_iris_train_X,
            self.local_iris_test_X,
            self.local_iris_train_Y,
            self.local_iris_test_Y,
        ) = sk_train_test_split(
            self.iris.select(iris_cols[:-1]).to_numpy(),
            self.iris.select(iris_cols[-1]).to_numpy().squeeze(),
            test_size=0.2,
            shuffle=False,
        )

    def test_linear_regression(self):
        from sklearn.linear_model import LinearRegression as SkLinearRegression

        lr = LinearRegression()
        sk_lr = SkLinearRegression()

        lr.fit(self.diabetes_train_X, self.diabetes_train_Y)
        sk_lr.fit(self.diabetes_local_train_X, self.diabetes_local_train_Y)
        remote_pred = lr.predict(self.diabetes_test_X).collect().fetch().to_numpy()
        local_pred = sk_lr.predict(self.diabetes_local_test_X)

        # Because of Machine epsilon, we round computation to 1e-10
        # and expect the non-zero values after calculation to be zero
        remote_pred = np.round(remote_pred, 10)
        local_pred = np.round(local_pred, 10)
        self.assertTrue(np.count_nonzero(remote_pred - local_pred) == 0)

    def test_logistic_regression(self):
        from sklearn.linear_model import LogisticRegression as SkLogisticRegression

        log_reg = LogisticRegression(multi_class="multinomial")
        sk_log_reg = SkLogisticRegression(max_iter=200)
        log_reg.fit(self.iris_train_X, self.iris_train_Y)
        sk_log_reg.fit(self.local_iris_train_X, self.local_iris_train_Y)

        pred = log_reg.predict(self.iris_test_X)
        local_pred = sk_log_reg.predict(self.local_iris_test_X)

    def tearDown(self) -> None:
        self.connection.close()
