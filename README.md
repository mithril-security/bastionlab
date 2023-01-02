<p align="center">
  <img src="docs/assets/logo.png" alt="BastionLab" width="200" height="200" />
</p>

<h1 align="center">Mithril Security – BastionLab</h1>

<h4 align="center">
  <a href="https://www.mithrilsecurity.io">Website</a> |
  <a href="https://bastionlab.readthedocs.io/en/latest/">Documentation</a> |
  <a href="https://discord.gg/TxEHagpWd4">Discord</a> |
  <a href="https://blog.mithrilsecurity.io/">Blog</a> |
  <a href="https://www.linkedin.com/company/mithril-security-company">LinkedIn</a> | 
  <a href="https://www.twitter.com/mithrilsecurity">Twitter</a>
</h4><br>

# 👋 Welcome to BastionLab! 

Where data owners and data scientists can securely collaborate without exposing data - opening the way to projects that were too risky to consider.

## ⚙️ What is BastionLab?

**BastionLab is a simple privacy framework for data science collaboration.** 

It acts like an **access control** solution, for data owners to protect the privacy of their datasets, **and stands as a guard**, to enforce that only privacy-friendly operations are allowed on the data and anonymized outputs are shown to the data scientist. 

- Data owners can let **external or internal data scientists explore and extract values from their datasets, according to a strict privacy policy they'll define in BastionLab**.
- Data scientists can **remotely run queries on data frames without seeing the original data or intermediary results**.

**BastionLab is an open-source project.** Our solution is coded in Rust 🦀 and uses Polars 🐻, a pandas-like library for data exploration.

## 🚀 Quick tour

You can go try out our [Quick tour](https://bastionlab.readthedocs.io/en/latest/docs/quick-tour/quick-tour/) in the documentation to discover BastionLab with a hands-on example using the famous Titanic dataset. 

But here’s a taste of what using BastionLab could look like 🍒

### Data owner's side
```py
# Load your dataset using polars.
>>> import polars as pl
>>> df = pl.read_csv("titanic.csv")

# Define a custom policy for your data.
# In this example, requests that aggregate at least 10 rows are safe.
# Other requests will be reviewed by the data owner.
>>> from bastionlab.polars.policy import Policy, Aggregation, Review
>>> policy = Policy(safe_zone=Aggregation(min_agg_size=10), unsafe_handling=Review())

# Upload your dataset to the server.
# Optionally anonymize sensitive columns.
# The server returns a remote object that can be used to query the dataset.
>>> from bastionlab import Connection
>>> with Connection("bastionlab.example.com") as client:
...     rdf = client.polars.send_df(df, policy=policy, sanitized_columns=["Name"])
...     rdf
...
FetchableLazyFrame(identifier=3a2d15c5-9f9d-4ced-9234-d9465050edb1)
```

### Data scientist's side
```py
# List the datasets made available by the data owner, select one and get a remote object.
>>> from bastionlab import Connection
>>> connection = Connection("localhost")
>>> all_remote_dfs = connection.client.polars.list_dfs()
>>> remote_df = all_remote_dfs[0]

# Run unsafe queries such as displaying the five first rows.
# According to the policy, unsafe queries require the data owner's approval.
>>> remote_df.head(5).collect().fetch()
Warning: non privacy-preserving queries necessitate data owner's approval.
Reason: Only 1 subrules matched but at least 2 are required.
Failed sub rules are:
Rule #1: Cannot fetch a DataFrame that does not aggregate at least 10 rows of the initial dataframe uploaded by the data owner.

A notification has been sent to the data owner. The request will be pending until the data owner accepts or denies it or until timeout seconds elapse.
The query has been accepted by the data owner.
shape: (5, 12)
┌─────────────┬──────────┬────────┬──────┬─────┬──────────────────┬─────────┬───────┬──────────┐
│ PassengerId ┆ Survived ┆ Pclass ┆ Name ┆ ... ┆ Ticket           ┆ Fare    ┆ Cabin ┆ Embarked │
│ ---         ┆ ---      ┆ ---    ┆ ---  ┆     ┆ ---              ┆ ---     ┆ ---   ┆ ---      │
│ i64         ┆ i64      ┆ i64    ┆ str  ┆     ┆ str              ┆ f64     ┆ str   ┆ str      │
╞═════════════╪══════════╪════════╪══════╪═════╪══════════════════╪═════════╪═══════╪══════════╡
│ 1           ┆ 0        ┆ 3      ┆ null ┆ ... ┆ A/5 21171        ┆ 7.25    ┆ null  ┆ S        │
├╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
│ 2           ┆ 1        ┆ 1      ┆ null ┆ ... ┆ PC 17599         ┆ 71.2833 ┆ C85   ┆ C        │
├╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
│ 3           ┆ 1        ┆ 3      ┆ null ┆ ... ┆ STON/O2. 3101282 ┆ 7.925   ┆ null  ┆ S        │
├╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
│ 4           ┆ 1        ┆ 1      ┆ null ┆ ... ┆ 113803           ┆ 53.1    ┆ C123  ┆ S        │
├╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
│ 5           ┆ 0        ┆ 3      ┆ null ┆ ... ┆ 373450           ┆ 8.05    ┆ null  ┆ S        │
└─────────────┴──────────┴────────┴──────┴─────┴──────────────────┴─────────┴───────┴──────────┘

# Run safe queries and get the result right away.
>>> (
... remote_df
... .select([pl.col("Pclass"), pl.col("Survived")])
... .groupby(pl.col("Pclass"))
... .agg(pl.col("Survived").mean())
... .sort("Survived", reverse=True)
... .collect()
... .fetch()
... )
shape: (3, 2)
┌────────┬──────────┐
│ Pclass ┆ Survived │
│ ---    ┆ ---      │
│ i64    ┆ f64      │
╞════════╪══════════╡
│ 1      ┆ 0.62963  │
├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
│ 2      ┆ 0.472826 │
├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
│ 3      ┆ 0.242363 │
└────────┴──────────┘
```

## 🗝️ Key features

- **Access control**: data owners can define an interactive privacy policy that will filter the data scientist queries. They do not have to open unrestricted access to their datasets anymore. 
- **Limited expressivity**: BastionLab limits the type of operations that can be executed by the data scientists to avoid arbitrary code execution.
- **Transparent remote access**: the data scientists never access the dataset directly. They only manipulate a local object that contains metadata to interact with a remotely hosted dataset. Calls can always be seen by data owners.

## 🙋 Getting help

- Go to our [Discord](https://discord.com/invite/TxEHagpWd4) #support channel
- Report bugs by [opening an issue on our BastionLab Github](https://github.com/mithril-security/bastionlab/issues)
- [Book a meeting](https://calendly.com/contact-mithril-security/15mins?month=2022-11) with us

## 🚨 Disclaimer

BastionLab is still in development. **Do not use it yet in a production workload.** We will audit our solution in the future to attest that it enforces the security standards of the market. 

## 📝 License

BastionLab is licensed under the Apache License, Version 2.0.

*Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.* 

*[See the License](http://www.apache.org/licenses/LICENSE-2.0) for the specific language governing permissions and limitations under the License.*
