from bastionlab import Connection, Identity, LinearRegression
import polars as pl

data_owner = Identity.create("data_owner")

client = Connection("localhost", identity=data_owner).client

# load dataset
train_df = pl.read_csv(
    "diabetes_train.csv",
    sep=" ",
    has_header=False,
    new_columns=["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"],
)
test_df = pl.read_csv(
    "diabetes_test.csv",
    has_header=False,
    new_columns=["target"],
)

train_rdf = client.send_df(train_df)
test_rdf = client.send_df(test_df)

# Perform linear regression on `diabetes` dataset.
model = client.train(train_rdf, test_rdf, 0.8, LinearRegression())
res = client.predict(model, [60, 2, 28.2, 112.0, 185, 113.8, 42.0, 4.0, 4.9836, 93])
print(res.fetch())
