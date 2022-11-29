# 👋 Welcome to BastionLab!

<font size="5"><span style="font-weight: 200">Where data owners and data scientists can securely collaborate without exposing data - opening the way to projects that were too risky to consider. </font></span>

## What is BastionLab?

**BastionLab is a data science framework to perform remote and secure Exploratory Data Analysis.**

- Data scientists can **remotely run queries on data frames without seeing the original data or intermediary results**.
- Data owners can let **external or internal data scientists explore and extract values from their datasets, according to the strict privacy policies they defined**.

This wasn’t possible until now for highly regulated fields like health, finance, or advertising. When collaborating remotely, data owners had to open their whole dataset, often through a Jupyter notebook. This was dangerous because too many operations were allowed and the data scientist had numerous ways to extract information from the remote infrastructure (print the whole database, save the dataset in the weights, etc).

BastionLab solves this problem by ensuring that only privacy-friendly operations are allowed on the data and aggregated outputs are shown to the data scientist. 

**BastionLab is an open-source project.** Our solution is coded in Rust and uses Polars, a pandas-like library for data exploration. You can check [the code on our GitHub](https://github.com/mithril-security/bastionlab/) and *(very soon)* our roadmap. 

We’ll update the documentation as new features come in, so dive in!

## Getting Started

- Follow our [“Quick Tour”](docs/quick-tour/quick-tour/) tutorial
- [Read](docs/concept-guides/threat_model/) about the technologies we use to ensure privacy
- Find [our benchmarks](docs/reference-guides/benchmarks/polars/) documenting BastionLab’s speed

## Getting Help

- Go to our [Discord](https://discord.com/invite/TxEHagpWd4) #support channel
- Report bugs by [opening an issue on our BastionLab Github](https://github.com/mithril-security/bastionlab/issues)
- [Book a meeting](https://calendly.com/contact-mithril-security/15mins?month=2022-11) with us

## How do we organize the documentation?

The security stakes with private data are high. BastionLab uses new technologies and we want to be sure that you understand what we protect you from and what we don’t protect you from.
<br>
Here’s a high-view of how we structure our documentation:

- **[Tutorials](docs/quick-tour/quick-tour/)** take you by the hand to install and run BastionLab. 
- **[Concept guides](docs/concept-guides/confidential_computing/)** discuss key topics and concepts at a high level. They provide useful background information and explanations.
- **| *Coming soon* | How-to guides** are recipes. They guide you through the steps involved in addressing key problems and use cases. They are more advanced than tutorials and assume some knowledge of how BastionLab works.
- **[Advanced guides](docs/reference-guides/deployment/on_premise/)** contain technical references for BastionLab’s machinery. They describe how it works and how to use it but assume you have a good understanding of key concepts.

## Who made BastionLab?
Mithril Security is a startup aiming to make privacy-friendly data science easy so data scientists and data owners can collaborate without friction. Our solutions apply Privacy Enhancing Technologies and security best practices, like [Remote Data Science](docs/concept-guides/remote_data_science/) and [Confidential Computing](docs/concept-guides/confidential_computing/).