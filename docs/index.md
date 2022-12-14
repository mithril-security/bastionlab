# 👋 Welcome to BastionLab!
________________________________________________________

<font size="5"><span style="font-weight: 200">Where data owners and data scientists can securely collaborate without exposing data - opening the way to projects that were too risky to consider.</font></span>

## What is BastionLab?
________________________________________________________

**BastionLab is a simple privacy framework for data science collaboration.** 

It acts like an **access control** solution, for data owners to protect the privacy of their datasets, **and stands as a guard**, to enforce that only privacy-friendly operations are allowed on the data and anonymized outputs are shown to the data scientist. 

- Data owners can let **external or internal data scientists explore and extract values from their datasets, according to a strict privacy policy they'll define in BastionLab**.
- Data scientists can **remotely run queries on data frames without seeing the original data or intermediary results**.

This wasn’t possible until now for highly regulated fields like health, finance, or advertising. When collaborating remotely, data owners had to open their whole dataset, often through a Jupyter notebook. This was dangerous because too many operations were allowed and the data scientist had numerous ways to extract information from the remote infrastructure (print the whole database, save the dataset in the weights, etc).

BastionLab solves this problem by ensuring that only privacy-friendly operations are allowed on the data and aggregated outputs are shown to the data scientist. 

**BastionLab is an open-source project.** Our solution is coded in Rust 🦀 and uses Polars 🐻, a pandas-like library for data exploration. You can check [the code on our GitHub](https://github.com/mithril-security/bastionlab/) and *(very soon)* our roadmap. 

We’ll update the documentation as new features come in, so dive in!

## Getting Started
________________________________________________________

- Follow our **[“Quick Tour”](docs/quick-tour/quick-tour.ipynb)** tutorial
- **[Read](docs/concept-guides/threat_model.md)** about the technologies we use to ensure privacy
- Find **[our benchmarks](docs/reference-guides/benchmarks/benchmarks.md)** documenting BastionLab’s speed

## Getting Help
________________________________________________________

- Go to our **[Discord](https://discord.com/invite/TxEHagpWd4) #support** channel
- Report bugs by **[opening an issue on our BastionLab Github](https://github.com/mithril-security/bastionlab/issues)**
- **[Book a meeting](https://calendly.com/contact-mithril-security/15mins?month=2022-11)** with us

## How do we organize the documentation?
____________________________________________

- **[Tutorials](docs/tutorials/installation.md)** take you by the hand to install and run BastionLab. We recommend you start with the **[Quick Tour](docs/quick-tour/quick-tour.ipynb)** and then move on to the other tutorials!  

- **[How-to guides](docs/use-cases/covid_use_case.ipynb)** are recipes. They guide you through the steps involved in addressing key problems and use cases. They are more advanced than tutorials and assume some knowledge of how BastionLab works.

- **[Concepts](docs/concept-guides/remote_data_science.md)** guides discuss key topics and concepts at a high level. They provide useful background information and explanations, especially on cybersecurity.

- **[Security](docs/concept-guides/threat_model.md)** guides contain technical information for security engineers. They explain the threat models and other cybersecurity topics required to audit BastionLab's security standards.

- **[Advanced](docs/resources/bastionlab/index.html)** guides contain technical references for BastionLab’s API machinery. They describe how it works and how to use it but assume you have a good understanding of key concepts. In the future, this section will also explain how to contribute to BastionLab. 

## Who made BastionLab?

[Mithril Security](https://www.mithrilsecurity.io/) is a startup aiming to make privacy-friendly data science easy so data scientists and data owners can collaborate without friction. Our solutions apply Privacy Enhancing Technologies and security best practices, like [Remote Data Science](docs/concept-guides/remote_data_science.md) and Confidential Computing.
