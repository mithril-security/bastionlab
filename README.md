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
- [Read](docs/docs/concept-guides/confidential_computing.md) about the technologies we use to ensure privacy
- Find [our benchmarks](docs/docs/reference-guides/benchmarks/polars.md) documenting BastionLab’s speed

## Getting Help
- Go to our [Discord](https://discord.com/invite/TxEHagpWd4) #support channel
- Report bugs by [opening an issue on our BastionLab Github](https://github.com/mithril-security/bastionlab/issues)
- [Book a meeting](https://calendly.com/contact-mithril-security/15mins?month=2022-11) with us

## How do we organize the documentation?

The security stakes with private data are high. BastionLab uses new technologies and we want to be sure that you understand what we protect you from and what we don’t protect you from.
<br>
Here’s a high-view of how we structure our documentation:

- **[Tutorials](docs/docs/quick-tour/quick-tour.ipynb)** take you by the hand to install and run BastionLab. You can get introduced to our “Quick Tour” first! 
- **[Concept guides](docs/docs/concept-guides/confidential_computing.md)** discuss key topics and concepts at a high level. They provide useful background information and explanations.
- **| *Coming soon* | How-to guides** are recipes. They guide you through the steps involved in addressing key problems and use cases. They are more advanced than tutorials and assume some knowledge of how BastionLab works.
- **[Reference guides](docs/docs/reference-guides/deployment/on_premise.md)** contain technical references for BastionLab’s machinery. They describe how it works and how to use it but assume you have a good understanding of key concepts.

## Who made BastionLab?
Mithril Security is a startup aiming to make privacy-friendly data science easy so data scientists and data owners can collaborate without friction. Our solutions apply Privacy Enhancing Technologies and security best practices, like [Remote Data Science]() and [Confidential Computing](docs/docs/concept-guides/confidential_computing.md).