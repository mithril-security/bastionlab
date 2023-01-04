# 👋 Welcome to BastionLab!
________________________________________________________

<font size="5"><span style="font-weight: 200">Where data owners and data scientists can securely collaborate without exposing data - opening the way to projects that were too risky to consider.</font></span>

## What is BastionLab?
________________________________________________________

**BastionLab is a simple privacy framework for data science collaboration, covering data exploration and AI traning.** 

It acts like an **access control solution**, for data owners to protect the privacy of their datasets, and **stands as a guard**, to enforce that only privacy-friendly operations are allowed on the data and anonymized outputs are shown to the data scientist. 

- Data owners can let external or internal data scientists explore and extract values from their datasets, according to a strict privacy policy they'll define in BastionLab.
- Data scientists can remotely run queries on data frames and train their models without seeing the original data or intermediary results.

This wasn’t possible until now for highly regulated fields like health, finance, or advertising. When collaborating remotely, data owners had to open their whole dataset, often through a Jupyter notebook. This was dangerous because too many operations were allowed and the data scientist had numerous ways to extract information from the remote infrastructure (print the whole database, save the dataset in the weights, etc).

BastionLab solves this problem by ensuring that no information is ever accessible locally to the data scientist. 

**BastionLab is an open-source project.** 
Our solution is coded in Rust 🦀, uses Polars 🐻, a pandas-like library for data exploration, and Torch 🔥, a popular library for AI training. 
We also have an option to set-up confidential computing 🔒, a hardware-based technology that ensures no one but the processor of the machine can see the data or the model.
 
You can check [the code on our GitHub](https://github.com/mithril-security/bastionlab/) and [our roadmap](https://mithril-security.notion.site/513af0ada2584e0f837776a7f6649ab4?v=cf664187c13149a4b667d9c0ae3ed1c0). 

We’ll update the documentation as new features come in, so dive in!

## Getting started
________________________________________________________

- Follow our [“Quick tour”](docs/quick-tour/quick-tour.ipynb) tutorial
- [Read](docs/security/threat_model_data_owner_owns_infrastructure.md) about the technologies we use to ensure privacy
- Find [our benchmarks](docs/advanced/benchmarks/polars_benchmarks.md) documenting BastionLab’s speed

## Getting help
________________________________________________________

- Go to our [Discord](https://discord.com/invite/TxEHagpWd4) *#support* channel
- Report bugs by [opening an issue on our BastionLab Github](https://github.com/mithril-security/bastionlab/issues)
- [Book a meeting](https://calendly.com/contact-mithril-security/15mins?month=2022-11) with us

## How do we organize the documentation?
____________________________________________

- [Tutorials](docs/tutorials/authentication.ipynb) take you by the hand to install and run BastionLab. We recommend you start with the **[Quick tour](docs/quick-tour/quick-tour.ipynb)** and then move on to the other tutorials!  

- [How-to guides](docs/how-to-guides/covid_cleaning_exploration.ipynb) are recipes. They guide you through the steps involved in addressing key problems and use cases. They are more advanced than tutorials and assume some knowledge of how BastionLab works.

- [Concepts](docs/concepts-guides/remote_data_science.md) guides discuss key topics and concepts at a high level. They provide useful background information and explanations, especially on cybersecurity.

- [API Reference](docs/resources/bastionlab/index.html) contains technical references for BastionLab’s API machinery. They describe how it works and how to use it but assume you have a good understanding of key concepts.

- [Security](docs/security/threat_model_data_owner_owns_infrastructure.md) guides contain technical information for security engineers. They explain the threat models and other cybersecurity topics required to audit BastionLab's security standards.

- [Advanced](docs/advanced/benchmarks/polars_benchmarks.md) guides are destined to developpers wanting to dive deep into BastionLab and eventually collaborate with us to the open-source code. We'll cover in the future exactly how to do so. 

## Who made BastionLab?

[Mithril Security](https://www.mithrilsecurity.io/) is a startup aiming to make privacy-friendly data science easy so data scientists and data owners can collaborate without friction. Our solutions apply Privacy Enhancing Technologies and security best practices, like [Remote Data Science](docs/concepts-guides/remote_data_science.md) and Confidential Computing.
