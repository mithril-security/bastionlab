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

**BastionLab is a data science framework to perform remote and secure Exploratory Data Analysis.** 

It acts like an **access control** solution, for data owners to protect the privacy of their datasets, **and stands as a guard**, to enforce that only privacy-friendly operations are allowed on the data and anonymized outputs are shown to an external data scientist. 

- Data owners can let **external or internal data scientists explore and extract values from their datasets, according to a strict privacy policy they'll define in BastionLab**.
- Data scientists can **remotely run queries on data frames without seeing the original data or intermediary results**.

**BastionLab is an open-source project.** Our solution is coded in Rust 🦀 and uses Polars 🐻, a pandas-like library for data exploration.

## 👌 Built to be easy and safe to use

Collaborating remotely and safely when it came to data science wasn’t possible until now for highly regulated fields like health, finance, or advertising. 

When they wanted to put their assets to good use, data owners had to open unrestricted access to their dataset, often through a Jupyter notebook. This was dangerous because too many operations were allowed and the data scientist had numerous ways to extract information from the remote infrastructure (print the whole database, save the dataset in the weights, etc). 

The other option was for the data scientist to send a Python script with all the operations to the server. Unless the data owner manually verified the whole for malicious code, data could still be easily exfiltrated. Even careful review doesn’t entirely remove the risk, due to human errors. On top of that, the whole process had to be repeated every time something had to be changed - writing, sending, verifying, executing the script - which was tedious and represented a huge organizational cost. Most of our clients explained they would just give up that option from the get go.

That is why we built BastionLab: we wanted a solution that would control remote access to datasets and allow privacy-friendly data science work. It would be interactive, fast, fit easily in the usual data science workflow of both data owners and data scientists, ensure privacy and be secure. 🔒

## 🚀 Quick Tour
You can go try out our [Quick Tour](https://github.com/mithril-security/bastionlab/tree/master/docs/docs/quick-tour) to discover BastionLab with a hands-on example using the famous Titanic dataset. 

But here’s a taste of what using BastionLab could look like 🍒
```py
import polars as pl
from bastionlab import Connection

df = pl.read_csv("titanic.csv")

with Connection("localhost") as client:
    remote_df = client.polars.send_df(
        df,
        sanitized_columns=["Name"],
    )
    print(remote_df.head(5).collect().fetch())
```

## 🗝️ Key Features

- **Access control**: data owners can define an interactive privacy policy that will filter the data scientist queries. They do not have to open unrestricted access to their datasets anymore. 
- **Limited expressivity**: BastionLab limits the type of operations that can be executed by the data scientists to the standard data science queries.
- **Transparent remote access**: the data scientists never access the dataset directly. They only manipulate a local object that contains metadata to interact with a remotely hosted dataset. Calls can always be seen by data owners.

## 🙋 Getting Help

- Go to our [Discord](https://discord.com/invite/TxEHagpWd4) #support channel
- Report bugs by [opening an issue on our BastionLab Github](https://github.com/mithril-security/bastionlab/issues)
- [Book a meeting](https://calendly.com/contact-mithril-security/15mins?month=2022-11) with us

## <img src="docs/assets/logo.png" alt="BastionLab" width="30" height="30" /> Who made BastionLab?
Mithril Security is a startup aiming to make privacy-friendly data science easy so data scientists and data owners can collaborate without friction. Our solutions apply Privacy Enhancing Technologies and security best practices, like [Remote Data Science](https://github.com/mithril-security/bastionlab/blob/master/docs/docs/concept-guides/remote_data_science.md) and [Confidential Computing](https://github.com/mithril-security/bastionlab/blob/master/docs/docs/concept-guides/confidential_computing.md).

## 🚨 License
BastionLab is still in development. We are licensed under the Apache License, Version 2.0. 

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 

[See the License](http://www.apache.org/licenses/LICENSE-2.0) for the specific language governing permissions and limitations under the License.
