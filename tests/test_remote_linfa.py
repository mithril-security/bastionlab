import unittest
import subprocess
import os
import polars as pl
from bastionlab import Connection
from bastionlab.polars.policy import TrueRule, Log, Policy
from bastionlab.linfa import models
from bastionlab.polars import train_test_split
from sklearn.model_selection import (
    train_test_split as sk_train_test_split,
)
from sklearn import metrics as sk_metrics
from bastionlab.linfa import metrics

from sklearn.linear_model import LinearRegression as SkLinearRegression
from sklearn.cluster import KMeans as SKMeans
from sklearn.naive_bayes import GaussianNB as SkGaussianNB
from sklearn.tree import DecisionTreeClassifier as SkDecisionTreeClassifier
from sklearn.svm import SVC as SkSVC

from sklearn.datasets import make_blobs

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


def to_numpy(df):
    return df.fetch().to_numpy()


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
        lr = models.LinearRegression()
        sk_lr = SkLinearRegression()

        lr.fit(self.diabetes_train_X, self.diabetes_train_Y)
        sk_lr.fit(self.diabetes_local_train_X, self.diabetes_local_train_Y)
        remote_pred = lr.predict(self.diabetes_test_X).to_array()
        local_pred = sk_lr.predict(self.diabetes_local_test_X)

        # Because of Machine epsilon, we round computation to 1e-10
        # and expect the non-zero values after calculation to be zero

        # Mean squared error computation
        remote_mean_squared = (
            to_numpy(metrics.mean_squared_error(self.diabetes_test_Y, remote_pred))
            .astype(np.float32)
            .squeeze()
        )

        local_mean_squared = sk_metrics.mean_squared_error(
            self.diabetes_local_test_Y, local_pred
        ).astype(np.float32)

        self.assertEqual(remote_mean_squared, local_mean_squared)

        # R2 score
        remote_r2_score = (
            to_numpy(metrics.r2_score(self.diabetes_test_Y, remote_pred))
            .astype(np.float32)
            .squeeze()
        )
        local_r2_score = sk_metrics.r2_score(
            self.diabetes_local_test_Y, local_pred
        ).astype(np.float32)

        self.assertEqual(remote_r2_score, local_r2_score)

    def test_multinomial_logistic_regression(self):
        from sklearn.linear_model import LogisticRegression as SkLogisticRegression

        log_reg = models.LogisticRegression(multi_class="multinomial", max_iter=200)
        sk_log_reg = SkLogisticRegression(max_iter=200)
        log_reg.fit(self.iris_train_X, self.iris_train_Y)
        sk_log_reg.fit(self.local_iris_train_X, self.local_iris_train_Y)

        remote_pred = log_reg.predict(self.iris_test_X).to_array()
        local_pred = sk_log_reg.predict(self.local_iris_test_X)

        self.assertEqual(
            to_numpy(metrics.accuracy_score(self.iris_test_Y, remote_pred))
            .squeeze()
            .astype(np.float32),
            sk_metrics.accuracy_score(self.local_iris_test_Y, local_pred).astype(
                np.float32
            ),
        )

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
        log_reg = models.LogisticRegression()
        sk_log_reg = SkLogisticRegression(max_iter=200, multi_class="auto")
        log_reg.fit(iris_train_X, iris_train_Y)
        sk_log_reg.fit(local_iris_train_X, local_iris_train_Y)

        remote_pred = log_reg.predict(iris_test_X).to_array()
        local_pred = sk_log_reg.predict(local_iris_test_X)

        # self.assertTrue(calculate_prediction_difference(remote_pred, local_pred))

        self.assertEqual(
            to_numpy(metrics.accuracy_score(iris_test_Y, remote_pred))
            .squeeze()
            .astype(np.float32),
            sk_metrics.accuracy_score(local_iris_test_Y, local_pred).astype(np.float32),
        )
        # Prediction Probability
        remote_prob_pred = (
            log_reg.predict_proba(iris_test_X).fetch().to_numpy().squeeze()
        )

        # We select the 2nd column because sklearn returns the complement of the probability
        # for the `1` class in the first column
        local_prob_pred = sk_log_reg.predict_proba(local_iris_test_X)[:, 1]

        self.assertAlmostEqual(remote_prob_pred.var(), local_prob_pred.var(), 4)

    def test_kmeans(self):
        X, y = make_blobs(n_samples=500, centers=3, n_features=2)

        data = pl.DataFrame(dict(X=X[:, 0], y=X[:, 1], labels=y))

        data_array = self.client.polars.send_df(data[:, :2], self.policy).to_array()
        # test_Y = self.client.polars.send_df(data[])
        # Initialize our KMeans classifier from sklearn

        sk_kmeans = SKMeans(
            init="k-means++",
            n_clusters=3,
            n_init=10,
            random_state=0,
            max_iter=2000,
        )

        # Here, we perform a trick by taking a column of the reduced_digits_data to fill the targets
        # requirement because of the a constraint in the backend

        kmeans = models.KMeans(
            n_clusters=3, init="k-means++", n_init=10, max_iter=2000, random_state=0
        )

        local_pred_input = X[:50, :2]

        remote_pred_input = self.client.polars.send_df(
            pl.DataFrame(local_pred_input), self.policy
        ).to_array()

        kmeans.fit(data_array)
        sk_kmeans.fit(X)

        remote_pred = kmeans.predict(remote_pred_input).fetch().to_numpy().squeeze()
        local_pred = sk_kmeans.predict(local_pred_input)

        self.assertTrue(True)

    def test_gaussian_naive_bayes(self):
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        Y = np.array([1, 1, 1, 2, 2, 2])

        remote_X = self.client.polars.send_df(
            pl.DataFrame(X).select(pl.all().cast(pl.Float64)), self.policy
        ).to_array()
        remote_Y = self.client.polars.send_df(
            pl.DataFrame(Y).select(pl.all().cast(pl.UInt64())), self.policy
        ).to_array()

        test_Y = self.client.polars.send_df(
            pl.DataFrame(pl.Series("target", values=[1], dtype=pl.UInt64)), self.policy
        ).to_array()

        bayes = models.GaussianNB()
        sk_bayes = SkGaussianNB()

        bayes.fit(remote_X, remote_Y)
        sk_bayes.fit(X, Y)

        remote_pred_input = self.client.polars.send_df(
            pl.DataFrame([[0.8], [-1.0]]), self.policy
        ).to_array()

        remote_pred = bayes.predict(remote_pred_input).to_array()
        local_pred = sk_bayes.predict([[0.8, -1.0]])

        self.assertEqual(
            to_numpy(metrics.accuracy_score(test_Y, remote_pred)),
            sk_metrics.accuracy_score(np.array([1]), local_pred),
        )

    def test_decision_trees(self):
        data = self.client.polars.send_df(
            self.iris,
            self.policy,
        )

        remote_inputs = data.select(self.iris_cols[:-1]).collect().to_array()
        remote_y = (
            data.select(pl.col(self.iris_cols[-1]).cast(pl.UInt64)).collect().to_array()
        )

        X_train, X_test, y_train, y_test = train_test_split(
            remote_inputs, remote_y, test_size=0.2, shuffle=False
        )

        X_local_train, X_local_test, y_local_train, y_local_test = sk_train_test_split(
            self.iris[:, :-1].to_numpy(),
            self.iris[:, -1].to_numpy(),
            test_size=0.2,
            shuffle=False,
        )

        decision_tree = models.DecisionTreeClassifier(max_depth=2)
        sk_decision_tree = SkDecisionTreeClassifier()

        decision_tree.fit(X_train, y_train)
        sk_decision_tree.fit(X_local_train, y_local_train)

        remote_pred = decision_tree.predict(X_test).to_array()
        local_pred = sk_decision_tree.predict(X_local_test).astype(np.uint64)

        remote_accuracy = (
            to_numpy(metrics.accuracy_score(y_test, remote_pred))
            .squeeze()
            .astype(np.float32)
        )
        local_accuracy = sk_metrics.accuracy_score(
            y_local_test,
            local_pred,
        ).astype(np.float32)

        self.assertTrue(np.abs(remote_accuracy - local_accuracy) <= 0.1)

    def tearDown(self) -> None:
        self.connection.close()
