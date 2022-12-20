<p align="center">
  <img src="https://github.com/mithril-security/bastionlab/raw/master/docs/assets/logo.png" alt="BastionLab" width="200" height="200" />
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

## 👌 Built to be easy and safe to use

Collaborating remotely and safely when it came to data science wasn’t possible until now for highly regulated fields like health, finance, or advertising. When they wanted to put their assets to good use, data owners had to open unrestricted access to their dataset, often through a Jupyter notebook. This was dangerous because too many operations were allowed and the data scientist had numerous ways to extract information from the remote infrastructure (print the whole database, save the dataset in the weights, etc). 

That is why we built BastionLab with the aim to ensure privacy while fitting easily in the whole data science workflow of both data owners and data scientists.

## 🚀 Quick tour

You can go try out our [Quick tour](https://github.com/mithril-security/bastionlab/tree/master/docs/docs/quick-tour/quick-tour.ipynb) in the documentation to discover BastionLab with a hands-on example using the famous Titanic dataset.

But here’s a taste of what using BastionLab could look like 🍒

### Data Owner's side
```py
from bastionlab import Connection
import polars as pl

df = pl.read_csv("titanic.csv")

with Connection("bastionlab.example.com") as client:
    client.polars.send_df(df)
```

### Data Scientist's side
```py
from bastionlab import Connection

with Connection("bastionlab.example.com") as client:
    all_remote_dfs = client.polars.list_dfs()
    remote_df = all_remote_dfs[0]
    remote_df.head(5).collect().fetch()
```

## 👀 What is this wheel

This wheel was made to deploy very easily BastionLab's server on a Google Colab/Jupyter Notebook environments.

**Please remember that while you will have most of the functionality of BastionLab, this wheel was not made to be used in production environments. If you want to personalize more the server and get the security features, it is recommanded to deploy the server yourself. Please refer to the documentation for more information.**

## 🗝️ Key Features

- **Access control**: data owners can define an interactive privacy policy that will filter the data scientist queries. They do not have to open unrestricted access to their datasets anymore. 
- **Limited expressivity**: BastionLab limits the type of operations that can be executed by the data scientists to avoid arbitrary code execution.
- **Transparent remote access**: the data scientists never access the dataset directly. They only manipulate a local object that contains metadata to interact with a remotely hosted dataset. Calls can always be seen by data owners.

## 🙋 Getting Help

- Go to our [Discord](https://discord.com/invite/TxEHagpWd4) #support channel
- Report bugs by [opening an issue on our BastionLab Github](https://github.com/mithril-security/bastionlab/issues)
- [Book a meeting](https://calendly.com/contact-mithril-security/15mins?month=2022-11) with us

## 🚨 Disclaimer

BastionLab is still in development. **Do not use it yet in a production workload.** We will audit our solution in the future to attest that it enforces the security standards of the market. 

## 📝 License

BastionLab is licensed under the Apache License, Version 2.0.

*Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.* 

*[See the License](http://www.apache.org/licenses/LICENSE-2.0) for the specific language governing permissions and limitations under the License.*