# Contributing to BastionAI
🎉 Hello there! thanks for taking the time to contribute to BastionAI! 🎉 

The following is a set of guidelines for contributing to [BastionAI](https://github.com/mithril-security/bastionai) project. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

#### Table Of Contents
[Code of Conduct](#code-of-conduct)

[What should I know before I get started?](#what-should-i-know-before-i-get-started)
  * [I only have a question!](#i-only-have-a-question)
  * [BastionAI Project](#bastionai-project)
  * [Useful Resources](#useful-resources)

[How Can I Contribute?](#how-can-i-contribute)
  * [Reporting Bugs](#reporting-bugs)
  * [Suggesting Enhancements](#suggesting-enhancements)
  * [Pull Requests](#pull-requests)
  * [Setting Your Local Development Environment](#setting-your-local-development-environment)
  * [Issue Tracker](#issue-tracker-tags)

## Code of Conduct

This project and everyone participating in it is governed by the [Mithril Security Code Of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [contact@mithrilsecurity.io](mailto:contact@mithrilsecurity.io).

## What should I know before I get started?

### I only have a question
If you have a question to ask or you want to open a discussion about BastionAI or confidential computing in general, we have a dedicated [Discord Community](https://discord.gg/TxEHagpWd4) in which all these kind of exchanges are more than welcome!

### BastionAI Project

**BastionAI🚀🔒** is a fast, easy-to-use confidential artificial intelligence (AI) platform for training AI models on private data.

With BastionAI, users can:

- Confidently fine-tune a model by sending data to the Cloud without data being exposed in clear.
- Securely train a model on datasets aggregated from multiple data sources, without any party having to show their data to the others.

The solution has two parts:
- A server which uses [tch-rs](), which contains Rust bindings for [libtorch]() Pytorch's C++ backend. It's used training AI models with privacy guarantees (Built with **Rust**).
- A client SDK to securely send datasets, and models to be trained on the server (Built with **Python**).

### BastionAI Project Structure.
```sh
BastionAI🚀🔐/
├─ Python Client /
├─ Rust Server/
│  ├─ BastionAI Learning/
│  ├─ BastionAI App/
│  ├─ BastionAI Common/

```
You can find more information about the **Roadmap** of the project [here](https://blog.mithrilsecurity.io/our-roadmap-at-mithril-security/#bastionai).

### Useful Resources

We highly encourage you to take a look at this resources for further information about BastionAI🚀🔐. 

It is also recommeneded to see the [examples](https://github.com/mithril-security/bastionai/tree/master/examples) that demonstrate how BastionAI works before submitting your first contribution. 

* [Documentation - BastionAI Official Documentation](https://bastionai.mithrilsecurity.io)
* [Blog - Mithril Security Blog](https://blog.mithrilsecurity.io/)
* [Article - Mithril Security Roadmap](https://blog.mithrilsecurity.io/our-roadmap-at-mithril-security/)
* [Notebooks and Python code - BastionAI Examples](https://github.com/mithril-security/bastionai/tree/master/examples)

## Contributing Code
This section presents the different options that you can follow in order to contribute to the  BastionAI🚀🔐 project. You can either **Report Bugs**, **Suggest Enhancements** or **Open Pull Requests**.

### Reporting Bugs
This section helps you through reporting Bugs for BastionAI. Following the guidelines helps the maintainers to understand your report, reproduce the bug and work on fixing at as soon as possible. 

> #### Important!
> Before reporting a bug, please take a look at the [existing issues](https://github.com/mithril-security/bastionai/issues). You may find that the bug has already been reported and that you don't need to create a new one.

#### How to report a bug? 
To report a Bug, you can either:
- Follow this [link](https://github.com/mithril-security/bastionai/issues/new?assignees=&labels=&template=bug-report.md&title=) and fill the report with the required information.
- In BastionAI github repository:
  * Go to `Issues` tab.
  * Click on `New Issue` button.
  * Choose the `Bug` option.
  * Fill the report with the required information.

#### How to submit a good Bug Report?
- Follow the Bug Report template as much as possible (You can add further details if needed).
- Use a clear and descriptive title.
- Describe the expected behavior, the one that's actually happening and how often does it reproduce.
- Describe the exact steps to reproduce the problem.
- Specify the versions of BastionAI Client and Server that produced the bug.
- Add any other relevant information about the context, your development environment (Operating system, Language version, Libtorch version, Platform)
- Attach screenshots, code snippets and any helpful resources.  

### Suggesting Enhancements 
This section guides you through suggesting enhancements for BastionAI project. You can suggest an enhancement by opening a **GitHub Issue**. 

> **Important!**
> Before opening an issue, please take a look at the [existing issues](https://github.com/mithril-security/bastionai/issues). You may find that the same suggestion has already been done and that you don't need to create a new one.

#### How to suggest an enhancement? 
To suggest enhamcement for BastionAI Project, you can either:

- Follow this [link](https://github.com/mithril-security/bastionai/issues/new/choose), choose the most relevant option and fill the report with the required information
- In BastionAI GitHub repository:
  * Go to `Issues` tab.
  * Click on `New Issue` button.
  * Choose the most relevant option.
  * Fill the description with the required information.

#### How to submit a good enhancement suggestion?
- Choose the most relevant issue type for your suggestion.
- Follow the provided template as much as possible.
- Use a clear and descriptive title.
- Add any other relevant information about the context, your development environment (Operating system, Language version ...)
- Attach screenshots, code snippets and any helpful resources. 

### Pull Requests
This section helps you through the process of opening a Pull Request and contributing with code to BastionAI Project!

#### How to open a pull request? 
In order to open a pull request:
- Go to BastionAI GitHub repository.
- Fork BastionAI project.
- [Setup your local development environment.](#setting-your-local-development-environment)
- Do your magic🚀🌠! and push your changes. 
- Open a Pull Request
- Fill the description with the required information.

#### How to submit a good pull request?
- Make sure your pull request solves an open issue or fixes a bug. If no related issue exists, please consider opening an issue first so that we can discuss your suggestions. 
- Follow the [style guidelines](#style-guidelines). 
- Make sure to use a clear and descriptive title.
- Follow the instructions in the pull request template.
- Provide as many relevant details as possible.
- Make sure to [link the related issues](https://docs.github.com/en/issues/tracking-your-work-with-issues/about-issues#efficient-communication) in the description.

While the prerequisites above must be satisfied prior to having your pull request reviewed, the reviewer(s) may ask you to complete additional work, tests, or other changes before your pull request can be ultimately accepted.

### Setting Your Local Development Environment
You can find detailed explanation of how to install BastionAI in your local machine in the [official documentation](https://bastionai.mithrilsecurity.io/getting-started/installation).

If you encounter any difficulties within that, don't hesitate to reach out to us through [Discord](https://discord.gg/TxEHagpWd4) and ask your questions. 


## Issue Tracker Tags

Issue type tags:

|     |     |
| --- | --- |
| question | Any questions about the project |
| bug | Something isn't working |
| enhancement | Improving performance, usability, consistency |
| docs | Documentation, tutorials, and example projects |
| new feature | Feature requests or pull request implementing a new feature |
| test | Improving unit test coverage, e2e test, CI or build |