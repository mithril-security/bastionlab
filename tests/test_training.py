from bastionlab.torch.remote_torch import RemoteDataset
from bastionlab.torch.optimizer_config import SGD, Adam
from bastionlab.torch.utils import TensorDataset
import polars as pl
import os
import torch
import subprocess
import logging
import unittest
from bastionlab import Connection
from bastionlab.polars.policy import (
    Policy,
    TrueRule,
    Log,
)
from bastionlab.polars import train_test_split

logging.basicConfig(level=logging.INFO)


def make_model(in_features: int, dtype: torch.dtype):
    class LinearRegression(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer1 = torch.nn.Linear(in_features, 1, dtype=dtype)

        def forward(self, tensor):
            return self.layer1(tensor)

    return LinearRegression


def get_covid_dataset():
    def runcmd(cmd, verbose=False, *args, **kwargs):
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True
        )
        std_out, std_err = process.communicate()
        if verbose:
            print(std_out.strip(), std_err)
        pass

    if not os.path.exists("covid.csv"):
        print("Downloading Covid Dataset")
        runcmd(
            'wget "https://raw.githubusercontent.com/rinbaruah/COVID_preconditions_Kaggle/master/Data/covid.csv"'
        )


def list_tensor_to_list_list(tensors):
    return [tensor.numpy().tolist() for tensor in tensors]


class TestingConnection(unittest.TestCase):
    def test_connection(self):
        connection = Connection("localhost")
        client = connection.client
        self.assertNotEqual(client, None)
        connection.close()

    def test_df_to_tensor_conv(self):
        connection = Connection("localhost")
        client = connection.client
        get_covid_dataset()
        df = pl.read_csv("covid.csv").limit(200)
        policy = Policy(safe_zone=TrueRule(), unsafe_handling=Log(), savable=True)
        rdf = client.polars.send_df(df, policy=policy, sanitized_columns=["Name"])
        rdf = rdf.drop(
            [
                "entry_date",
                "date_symptoms",
                "date_died",
                "patient_type",
                "sex",
                "id",
                "date",
                "intubed",
                "pregnancy",
                "contact_other_covid",
                "icu",
            ]
        )
        label_col = "covid_res"
        cols = list(filter(lambda a: a != label_col, rdf.columns))

        rdf = rdf.with_column(
            pl.when(pl.col(label_col) == 2).then(1).otherwise(0).alias(label_col)
        )
        rdf = rdf.collect()

        inputs = rdf.select(cols).collect().to_array()
        labels = rdf.select(label_col).collect().to_array()

        train_inputs, test_inputs, train_labels, test_labels = train_test_split(
            inputs,
            labels,
            test_size=0.2,
            shuffle=True,
        )

        train_inputs = train_inputs.to_tensor().to(torch.float32)
        train_labels = train_labels.to_tensor().to(torch.float32)
        test_inputs = test_inputs.to_tensor().to(torch.float32)
        test_labels = test_labels.to_tensor().to(torch.float32)

        in_features = train_inputs.shape[-1]
        dtype = train_inputs.dtype
        model = make_model(in_features, dtype)()

        train_dataset = client.torch.RemoteDataset(
            inputs=[train_inputs], labels=train_labels
        )
        test_dataset = client.torch.RemoteDataset(
            inputs=[test_inputs], labels=test_labels
        )

        self.assertNotEqual(train_dataset.identifier, "")
        self.assertNotEqual(test_dataset.identifier, "")

        remote_learner = connection.client.torch.RemoteLearner(
            model,
            train_dataset,
            max_batch_size=2,
            loss="cross_entropy",
            optimizer=Adam(lr=5e-5),
            model_name="LinearRegression",
        )

        remote_learner.fit(nb_epochs=1)

        remote_learner.test(test_dataset)

        fetched_model = remote_learner.get_model()

        self.assertEqual(fetched_model, model)

        connection.close()

    def test_sgd_linear_regression(self):
        connection = Connection("localhost")
        client = connection.client
        X = torch.tensor([[0.0], [1.0], [0.5], [0.2]])
        Y = torch.tensor([[0.0], [2.0], [1.0], [0.4]])
        train_dataset = TensorDataset([X], Y)

        X = torch.tensor([[0.1], [-1.0]])
        Y = torch.tensor([[0.2], [-2.0]])
        test_dataset = TensorDataset([X], Y)

        train_dataset = client.torch.RemoteDataset(
            train_dataset,
            name="1D Linear Regression",
            description="Dummy 1D Linear Regression Dataset (param is 2)",
        )
        test_dataset = client.torch.RemoteDataset(
            test_dataset,
            name="1D Linear Regression",
            description="Dummy 1D Linear Regression Dataset (param is 2)",
        )

        model = make_model(in_features=X.shape[-1], dtype=X.dtype)()

        remote_learner = client.torch.RemoteLearner(
            model,
            train_dataset,
            max_batch_size=2,
            loss="l2",
            optimizer=SGD(lr=0.1, momentum=0.9),
            model_name="Linear 1x1",
            model_description="1D Linear Regression Model",
        )
        remote_learner.fit(
            nb_epochs=1,
        )

        fetched_model = remote_learner.get_model()

        remote_learner.test(test_dataset, metric_eps=20.0)
        self.assertEqual(fetched_model, model)

        client.torch.delete_dataset(train_dataset)
        client.torch.delete_dataset(test_dataset)
        connection.close()

    def test_dp_sgd_linear_regression(self):
        connection = Connection("localhost")
        client = connection.client
        X = torch.tensor([[0.0], [1.0], [0.5], [0.2]])
        Y = torch.tensor([[0.0], [2.0], [1.0], [0.4]])
        train_dataset = TensorDataset([X], Y)

        X = torch.tensor([[0.1], [-1.0]])
        Y = torch.tensor([[0.2], [-2.0]])
        test_dataset = TensorDataset([X], Y)

        train_dataset = client.torch.RemoteDataset(
            train_dataset,
            name="1D Linear Regression",
            description="Dummy 1D Linear Regression Dataset (param is 2)",
            privacy_limit=8001.1,
        )
        test_dataset = client.torch.RemoteDataset(
            test_dataset,
            name="1D Linear Regression",
            description="Dummy 1D Linear Regression Dataset (param is 2)",
            privacy_limit=8001.1,
        )

        model = make_model(in_features=X.shape[-1], dtype=X.dtype)()

        remote_learner = client.torch.RemoteLearner(
            model,
            train_dataset,
            max_batch_size=2,
            loss="l2",
            optimizer=SGD(lr=0.1, momentum=0.9),
            model_name="Linear 1x1",
            model_description="1D Linear Regression Model",
            expand=True,
        )
        remote_learner.fit(nb_epochs=1, eps=5.0)

        fetched_model = remote_learner.get_model()

        remote_learner.test(test_dataset)
        self.assertEqual(fetched_model, model)

        client.torch.delete_dataset(train_dataset)
        client.torch.delete_dataset(test_dataset)

        connection.close()

    def test_available_datasets(self):
        connection = Connection("localhost")
        client = connection.client
        X = torch.tensor([[0.0], [1.0], [0.5], [0.2]])
        Y = torch.tensor([[0.0], [2.0], [1.0], [0.4]])
        train_dataset = TensorDataset([X], Y)

        X = torch.tensor([[0.1], [-1.0]])
        Y = torch.tensor([[0.2], [-2.0]])
        test_dataset = TensorDataset([X], Y)

        count = len(client.torch.get_available_datasets())
        train_dataset = client.torch.RemoteDataset(
            train_dataset,
            name="1D Linear Regression",
            description="Dummy 1D Linear Regression Dataset (param is 2)",
            privacy_limit=8001.1,
        )
        test_dataset = client.torch.RemoteDataset(
            test_dataset,
            name="1D Linear Regression",
            description="Dummy 1D Linear Regression Dataset (param is 2)",
            privacy_limit=8001.1,
        )

        after_count = len(client.torch.get_available_datasets())

        self.assertEqual(after_count, count + 2)

        connection.close()

    def test_delete_datasets(self):
        connection = Connection("localhost")
        client = connection.client
        X = torch.tensor([[0.0], [1.0], [0.5], [0.2]])
        Y = torch.tensor([[0.0], [2.0], [1.0], [0.4]])
        train_dataset = TensorDataset([X], Y)

        train_dataset = client.torch.RemoteDataset(
            train_dataset,
            name="1D Linear Regression",
            description="Dummy 1D Linear Regression Dataset (param is 2)",
            privacy_limit=8001.1,
        )

        count = len(client.torch.get_available_datasets())
        client.torch.delete_dataset(train_dataset)
        after_count = len(client.torch.get_available_datasets())

        self.assertEqual(after_count, count - 1)
        connection.close()

    def test_fetch_dataset(self):
        connection = Connection("localhost")
        client = connection.client
        X = torch.tensor([[0.0], [1.0], [0.5], [0.2]])
        Y = torch.tensor([[0.0], [2.0], [1.0], [0.4]])
        tensor_train = TensorDataset([X], Y)

        train_dataset = client.torch.RemoteDataset(
            tensor_train,
            name="1D Linear Regression",
            description="Dummy 1D Linear Regression Dataset (param is 2)",
            privacy_limit=8001.1,
        )

        fetched_tensor_train = client.torch.fetch_dataset(train_dataset)
        self.assertListEqual(
            list_tensor_to_list_list(fetched_tensor_train.columns),
            list_tensor_to_list_list(tensor_train.columns),
        )

        connection.close()


if __name__ == "__main__":
    unittest.main()
