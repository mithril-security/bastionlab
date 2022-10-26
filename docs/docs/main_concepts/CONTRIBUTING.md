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

[Style Guidelines](#style-guidelines)
  * [Git Commit Messages](#git-commit-messages)
  * [Linting and Formatting](#linting-and-formatting)
  * [Pre-commit hook](#pre-commit-hook)

[Additional Notes](#additional-notes)
  * [Issue and Pull Request Labels](#issue-and-pull-request-labels)

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

You can find more information about the **Roadmap** of the project [here](https://blog.mithrilsecurity.io/our-roadmap-at-mithril-security/#bastionai).

### Useful Resources

We highly encourage you to take a look at this resources for further information about BastionAI🚀🔐. 

It is also recommeneded to see the [examples](https://github.com/mithril-security/bastionai/tree/master/examples) that demonstrate how BastionAI works before submitting your first contribution. 

* [Documentation - BastionAI Official Documentation](https://bastionai.mithrilsecurity.io)
* [Blog - Mithril Security Blog](https://blog.mithrilsecurity.io/)
* [Article - Mithril Security Roadmap](https://blog.mithrilsecurity.io/our-roadmap-at-mithril-security/)
* [Notebooks - BastionAI Examples](https://github.com/mithril-security/bastionai/tree/master/examples)

## Contributing Code

## Contributing Documentation