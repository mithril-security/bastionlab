from bastionlab.torch.remote_torch import RemoteDataset
from bastionlab.torch.optimizer_config import Adam
import polars as pl
import torch
from sklearn.model_selection import train_test_split as sk_train_test_split
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
            self.layer1 = torch.nn.Linear(in_features, 2, dtype=dtype)

        def forward(self, tensor):
            return self.layer1(tensor)

    return LinearRegression


class TestingConnection(unittest.TestCase):
    def test_connection(self):
        connection = Connection("localhost")
        client = connection.client
        self.assertNotEqual(client, None)
        connection.close()

    def test_df_to_tensor_conv(self):
        connection = Connection("localhost")
        client = connection.client
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
        cols = list(filter(lambda a: a != label_col, rdf.column_names))

        rdf = rdf.with_column(
            pl.when(pl.col(label_col) == 2).then(1).otherwise(0).alias(label_col)
        )
        rdf = rdf.collect()

        inputs = rdf.select(cols).collect()
        labels = rdf.select(label_col).collect()

        train_inputs, test_inputs, train_labels, test_labels = train_test_split(
            inputs,
            labels,
            test_size=0.2,
            shuffle=True,
        )

        train_inputs = train_inputs.to_tensor().to(torch.float32)
        train_labels = train_labels.to_tensor()
        test_inputs = test_inputs.to_tensor().to(torch.float32)
        test_labels = test_labels.to_tensor()

        in_features = train_inputs.shape[-1]
        dtype = train_inputs.dtype
        model = make_model(in_features, dtype)()

        train_dataset = RemoteDataset(inputs=[train_inputs], labels=train_labels)
        test_dataset = RemoteDataset(inputs=[test_inputs], labels=test_labels)

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


if __name__ == "__main__":

    unittest.main()
