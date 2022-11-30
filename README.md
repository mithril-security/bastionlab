<p align="center">
  <img src="docs/assets/logo.png" alt="BastionLab" width="200" height="200" />
</p>

<h1 align="center">Mithril Security – BastionLab</h1>

<h4 align="center">
  <a href="https://www.mithrilsecurity.io">Website</a> |
  <a href="https://bastionlab.readthedocs.io/en/latest/">Documentation</a> |
  <a href="https://blog.mithrilsecurity.io/">Blog</a> |
  <a href="https://www.linkedin.com/company/mithril-security-company">LinkedIn</a> | 
  <a href="https://www.twitter.com/mithrilsecurity">Twitter</a> | 
  <a href="https://discord.gg/TxEHagpWd4">Discord</a>
</h4><br>


# 👋 Welcome to BastionLab!

Where data owners and data scientists can securely collaborate without exposing data - opening the way to projects that were too risky to consider. 

## What is BastionLab?

**BastionLab is a data science framework to perform remote and secure Exploratory Data Analysis.**

- Data scientists can **remotely run queries on data frames without seeing the original data or intermediary results**.
- Data owners can let **external or internal data scientists explore and extract values from their datasets, according to the strict privacy policies they defined**.

This wasn’t possible until now for highly regulated fields like health, finance, or advertising. When collaborating remotely, data owners had to open their whole dataset, often through a Jupyter notebook. This was dangerous because too many operations were allowed and the data scientist had numerous ways to extract information from the remote infrastructure (print the whole database, save the dataset in the weights, etc).

BastionLab solves this problem by ensuring that only privacy-friendly operations are allowed on the data and anonymized outputs are shown to the data scientist. 

**BastionLab is an open-source project.** Our solution is coded in Rust and uses Polars, a pandas-like library for data exploration.

>  **Disclaimer**: BastionAI, our fortified learning framework using TEEs will be incorporated in the broader offers of BastionLab, a holistic secure data science toolkit. While waiting for the merge, you can still use BastionAI under the folder bastionai.

## Getting Started

- Follow our [“Quick Tour”](docs/docs/quick-tour/quick-tour.ipynb) tutorial
- [Read](docs/docs/concept-guides/threat_model.md) about the technologies we use to ensure privacy
- Find [our benchmarks](docs/docs/reference-guides/benchmarks/polars.md) documenting BastionLab’s speed

## Getting Help
- Go to our [Discord](https://discord.com/invite/TxEHagpWd4) #support channel
- Report bugs by [opening an issue on our BastionLab Github](https://github.com/mithril-security/bastionlab/issues)
- [Book a meeting](https://calendly.com/contact-mithril-security/15mins?month=2022-11) with us

## Who made BastionLab?
Mithril Security is a startup aiming to make privacy-friendly data science easy so data scientists and data owners can collaborate without friction. Our solutions apply Privacy Enhancing Technologies and security best practices, like [Remote Data Science]() and [Confidential Computing]().
