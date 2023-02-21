import unittest
import subprocess
import os
import polars as pl
from bastionlab import Connection
from bastionlab.polars.policy import TrueRule, Log, Policy
from bastionlab.linfa import LinearRegression, LogisticRegression, cross_validate
from bastionlab.polars import train_test_split
from sklearn.model_selection import (
    train_test_split as sk_train_test_split,
    cross_validate as sk_cross_validate,
    cross_val_score,
)
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression as SkLinearRegression

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

    diabetes_train = "diabetes_train.csv"
    diabetes_targets = "diabetes_targets.csv"

    # Iris Dataset
    iris_dataset = "iris.csv"
    iris_dataset_link = "https://raw.githubusercontent.com/scikit-learn/scikit-learn/98cf537f5c538fdbc9d27b851cf03ce7611b8a48/sklearn/datasets/data/iris.csv"

    # Digits dataset for kmeans test
    digits_dataset = "digits.csv"
    digits_dataset_link = "https://github.com/scikit-learn/scikit-learn/raw/main/sklearn/datasets/data/digits.csv.gz"
    cmds = [
        f"wget -O {diabetes_train}.gz {diabetes_train_link}",
        f"wget -O {diabetes_targets}.gz {diabetes_targets_link}",
        f"wget -O {iris_dataset} {iris_dataset_link}",
        f"wget -O {digits_dataset}.gz {digits_dataset_link}",
        f"gzip -d {diabetes_train}.gz",
        f"gzip -d {diabetes_targets}.gz",
        f"gzip -d {digits_dataset}.gz",
    ]

    for cmd in cmds:
        cmd_parts = cmd.split(" ")
        filename = cmd_parts[2].split(".gz")[0]
        if not os.path.exists(filename):
            if cmd_parts[0] == "wget":
                print(f"Downloading Dataset: {filename}")
            runcmd(cmd)


def calculate_prediction_difference(remote_pred, local_pred):
    remote_pred = remote_pred.astype(float)
    local_pred = local_pred.astype(float)
    remote_pred = np.round(remote_pred, 10)
    local_pred = np.round(local_pred, 10)
    return np.count_nonzero(remote_pred - local_pred) < 3


def reshuffle_dataframe(dataframe: pl.DataFrame) -> pl.DataFrame:
    array = dataframe.to_numpy()
    np.random.shuffle(array)
    return pl.DataFrame(array)


class TestingRemoteLinfa(unittest.TestCase):
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
        self.iris = reshuffle_dataframe(self.iris)

        self.connection = Connection("localhost")

        self.policy = Policy(TrueRule(), Log(), False)

        self.client = self.connection.client
        self.diabetes_train_rdf = self.client.polars.send_df(
            self.diabetes_train, self.policy
        )

        self.diabetes_train_array = (
            self.diabetes_train_rdf.select(pl.all().cast(pl.Float64))
            .collect()
            .to_array()
        )

        self.diabetes_targets_rdf = self.client.polars.send_df(
            self.diabetes_targets, self.policy
        ).collect()

        self.iris_cols = self.iris.columns
        self.iris_train_array = self.client.polars.send_df(
            self.iris.select(self.iris_cols[:-1]), self.policy
        ).to_array()

        self.iris_targets_array = self.client.polars.send_df(
            self.iris.select(pl.col(self.iris_cols[-1]).cast(pl.UInt64)), self.policy
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

        (
            self.local_iris_train_X,
            self.local_iris_test_X,
            self.local_iris_train_Y,
            self.local_iris_test_Y,
        ) = sk_train_test_split(
            self.iris.select(self.iris_cols[:-1]).to_numpy(),
            self.iris.select(self.iris_cols[-1]).to_numpy().squeeze(),
            test_size=0.2,
            shuffle=False,
        )

    def test_linear_regression(self):
        lr = LinearRegression()
        sk_lr = SkLinearRegression()

        lr.fit(self.diabetes_train_X, self.diabetes_train_Y)
        sk_lr.fit(self.diabetes_local_train_X, self.diabetes_local_train_Y)
        remote_pred = lr.predict(self.diabetes_test_X).collect().fetch().to_numpy()
        local_pred = sk_lr.predict(self.diabetes_local_test_X)

        # Because of Machine epsilon, we round computation to 1e-10
        # and expect the non-zero values after calculation to be zero
        self.assertTrue(calculate_prediction_difference(remote_pred, local_pred))

    def test_linear_regression_accuracy(self):
        """
        To test for the accuracy of the linear regression model on BastionLab,
        we will first shuffle the DataFrame to reduce overfitting in our model."""

        # Join the input and target to make sure we shuffle rightly
        diabetes = pl.concat(
            [self.diabetes_train, self.diabetes_targets], how="horizontal"
        )
        # diabetes = reshuffle_dataframe(diabetes)

        # Split the dataframe back into inputs and targets
        columns = diabetes.columns

        # Targets is the last column
        diabetes_train = diabetes.select(pl.col(columns[:-1]).cast(pl.Float64))
        diabetes_targets = diabetes.select(columns[-1])
        diabetes_train_array = self.client.polars.send_df(
            diabetes_train, self.policy
        ).to_array()
        diabetes_targets_array = self.client.polars.send_df(
            diabetes_targets, self.policy
        ).to_array()

        (
            diabetes_train_X,
            diabetes_test_X,
            diabetes_train_Y,
            diabetes_test_Y,
        ) = train_test_split(
            diabetes_train_array,
            diabetes_targets_array,
            test_size=0.30,
            shuffle=False,
        )

        (
            diabetes_local_train_X,
            diabetes_local_test_X,
            diabetes_local_train_Y,
            diabetes_local_test_Y,
        ) = sk_train_test_split(
            diabetes_train.to_numpy(),
            diabetes_targets.to_numpy(),
            test_size=0.30,
            shuffle=False,
        )

        lr = LinearRegression()
        sk_lr = SkLinearRegression()

        lr.fit(diabetes_train_X, diabetes_train_Y)
        sk_lr.fit(diabetes_local_train_X, diabetes_local_train_Y)

        remote_pred = lr.predict(diabetes_test_X)
        local_pred = sk_lr.predict(diabetes_local_test_X)

        # TODO: Write test cases for cross_validation

        scores = [
            dict(remote="r2", local="r2"),
            dict(remote="max_error", local="max_error"),
            dict(remote="mean_absolute_error", local="neg_mean_absolute_error"),
            dict(remote="explained_variance", local="explained_variance"),
            dict(remote="mean_squared_log_error", local="neg_mean_squared_log_error"),
            dict(remote="mean_squared_error", local="neg_mean_squared_error"),
            dict(remote="median_absolute_error", local="neg_median_absolute_error"),
        ]
        for score in scores:
            # Cross_validate scores
            remote_scores = (
                cross_validate(
                    lr,
                    remote_pred.to_array(),
                    diabetes_test_Y,
                    cv=20,
                    scoring=score["remote"],
                )
                .fetch()
                .to_numpy()
                .squeeze()
            )

            local_scores = cross_val_score(
                sk_lr,
                local_pred,
                diabetes_local_test_Y,
                cv=20,
                scoring=score["local"],
            ).mean()

            ## TODO: Add the best relationship checker to assert on

    def test_multinomial_logistic_regression(self):
        from sklearn.linear_model import LogisticRegression as SkLogisticRegression

        log_reg = LogisticRegression(multi_class="multinomial", max_iterations=200)
        sk_log_reg = SkLogisticRegression(max_iter=200)
        log_reg.fit(self.iris_train_X, self.iris_train_Y)
        sk_log_reg.fit(self.local_iris_train_X, self.local_iris_train_Y)

        remote_pred = (
            log_reg.predict(self.iris_test_X).collect().fetch().to_numpy().squeeze()
        )
        local_pred = sk_log_reg.predict(self.local_iris_test_X)

        self.assertTrue(calculate_prediction_difference(remote_pred, local_pred))

        remote_prob_pred = (
            log_reg.predict_proba(self.iris_test_X).fetch().to_numpy().squeeze()
        )
        local_prob_pred = sk_log_reg.predict_proba(self.local_iris_test_X)

        self.assertAlmostEqual(remote_prob_pred.var(), local_prob_pred.var(), places=4)

    def test_multinomial_logistic_regression_accuracy(self):
        iris = reshuffle_dataframe(self.iris)

        columns = iris.columns

        iris_train = iris.select(columns[:-1])
        iris_targets = iris.select(columns[-1]).select(pl.all().cast(pl.UInt64))

        iris_train_array = self.client.polars.send_df(
            iris_train, self.policy
        ).to_array()
        iris_targets_array = self.client.polars.send_df(
            iris_targets, self.policy
        ).to_array()

    def test_binomial_logistic_regression(self):
        from sklearn.linear_model import LogisticRegression as SkLogisticRegression

        # For binomial, we can reduce the iris dataset to just 2 classes
        # `0` and `1` instead of `0, 1, 2`
        target_col = self.iris.columns[-1]

        self.iris = reshuffle_dataframe(self.iris)

        zero_targets = self.iris.select(pl.all()).filter(pl.col(target_col) == 0.0)
        ones_targets = self.iris.select(pl.all()).filter(pl.col(target_col) == 1.0)

        filtered = zero_targets.vstack(ones_targets)

        iris_train_array = self.client.polars.send_df(
            filtered.select(self.iris_cols[:-1]), Policy(TrueRule(), Log(), False)
        ).to_array()

        iris_targets_array = self.client.polars.send_df(
            filtered.select(pl.col(self.iris_cols[-1]).cast(pl.UInt64)),
            Policy(TrueRule(), Log(), False),
        ).to_array()

        (iris_train_X, iris_test_X, iris_train_Y, iris_test_Y,) = train_test_split(
            iris_train_array, iris_targets_array, test_size=0.2, shuffle=False
        )

        (
            local_iris_train_X,
            local_iris_test_X,
            local_iris_train_Y,
            local_iris_test_Y,
        ) = sk_train_test_split(
            filtered.select(self.iris_cols[:-1]).to_numpy(),
            filtered.select(self.iris_cols[-1]).to_numpy().squeeze(),
            test_size=0.2,
            shuffle=False,
        )
        log_reg = LogisticRegression()
        sk_log_reg = SkLogisticRegression(max_iter=200, multi_class="auto")
        log_reg.fit(iris_train_X, iris_train_Y)
        sk_log_reg.fit(local_iris_train_X, local_iris_train_Y)

        remote_pred = (
            log_reg.predict(iris_test_X).collect().fetch().to_numpy().squeeze()
        )
        local_pred = sk_log_reg.predict(local_iris_test_X)

        self.assertTrue(calculate_prediction_difference(remote_pred, local_pred))

        # Prediction Probability
        remote_prob_pred = (
            log_reg.predict_proba(iris_test_X).fetch().to_numpy().squeeze()
        )

        # We select the 2nd column because sklearn returns the complement of the probability
        # for the `1` class in the first column
        local_prob_pred = sk_log_reg.predict_proba(local_iris_test_X)[:, 1]

        self.assertAlmostEqual(remote_prob_pred.var(), local_prob_pred.var(), 4)

    def tearDown(self) -> None:
        self.connection.close()
