<style>
    .header {
        color: red;
    }
    .bastionai{
        text-decoration: none;
        font-weight: 500; 
        color: #fff; 
        font-size: 16px;
        transition: all 0.1s ease;
    }
    .bastionai:hover {
        text-decoration: none;
        cursor: pointer;
        color: #C9D8E4;
    }
    .comparison {
        font-size: 20px;
    }
    .feature-card {
    background: #384343;
    margin: 8px 8px 8px 0;
    display:flex; 
    flex-direction: column; 
    padding: 2px 8px;  
    border-radius: 4px;
    }
    .feature-grid {
        display: grid; 
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
        margin-bottom: 15px;
    }
    .feature-title {
        font-weight: 600; font-size: 17px; margin: 8px 0;
    }
</style>
# What is BastionAI?
<a href="https://github.com/mithril-security/bastionai.git" style="text-decoration: none; font-weight: 500; color: #fff; font-size: 18px">BastionAI🚀🔒</a> is a fast, easy-to-use confidential artificial intelligence (AI) platform for training AI models on private data.

With BastionAI, users can:

- Confidently fine-tune a model by sending data to the Cloud without data being exposed in clear.
- Securely train a model on datasets aggregated from multiple data sources, without any party having to show their data to the others.

BastionAI uses state-of-the art Privacy Enhancing Technologies, such as <a href="https://blog.mithrilsecurity.io/confidential-computing-explained-part-1-introduction/">Confidential Computing</a> and <a href="https://en.wikipedia.org/wiki/Differential_privacy">Differential Privacy</a> (DP), to protect data from being accessed by third parties. 

This framework for privacy-preserving deep learning leveraging secure enclaves and DP. The library departs from traditional decentralized FL and
proposes a **”fortified learning”** approach, where computations are centralized in a
trusted execution environment (TEE).

Fortified learning allows for faster training,
reasonable DP noise for the same budget, and simplifies deployment as each participating node only needs a lightweight interface to check the security features
of the remote enclave.

## Features
<div class="feature-grid">
    <div class="feature-card">
        <h4 class="feature-title">🔐 Confidentiality made easy</h4>
        <p style="font-size: 15px">Easily train state-of-the-art deep learning models with confidentiality thanks to the use of TEEs and Differential Privacy.</p>
    </div>
    <div style="background:#114455;" class="feature-card"> 
        <h4 class="feature-title">🪟 Transparency</h4>
        <p style="font-size: 15px">Provide guarantees to third parties, for instance clients or regulators, that you are indeed providing data protection, through code attestation.</p>
    </div>
    <div style="background:#792D16;" class="feature-card"> 
        <h4 class="feature-title">⚙️ Extensible usage</h4>
        <p style="font-size: 15px">Explore different scenarios from confidential multi party training of ResNets for cancer detection, to fine-tuning of GPT models on confidential text, such as emails or documents.</p>
    </div>
</div>

## Why are we building BastionAI🚀🔒?
Today, most AI tools offer no privacy by design mechanisms: when multiple parties pool data to train an AI model  (e.g. for healthcare purposes), sensitive data can be exposed to multiple third parties, posing security and privacy threats.

Privacy Enhancing Technologies (PETs) have emerged as an answer to these issues. Differential Privacy, for instance, provides strong guarantees through mathematics on the amount of information leaked by a computation whose output is publicly disclosed. Confidential Computing with secure enclaves makes it possible for Data Owners to have third parties manipulate their data in the Cloud, without exposing them in clear to anyone else.

However promising these technologies might be, finding secure and **easy-to-use** solutions still remains difficult. This is why we have built <a class="bastionai">BastionAI🚀🔒</a>, a confidential AI training framework, to make these technologies more accessible.

In addition, democratizng AI privacy solutions is also a contributing factor to why we decided to develop this solution. A solution developed by data scientists for data scientists is primordial to open-source community, having the number one focus of being an **easy-to-use** AI training platform.


# Comparisons

<p class="comparison">
<a href="https://github.com/mithril-security/bastionai.git" class="bastionai">BastionAI🚀🔒</a> versus Centralized Training.
</p>
Ordinarily, without the use of Trusted Execution Environments (TEEs), centralized training of deep learning models would have an elevated security threat level if training occurs on private data. Cloud providers, model owners, and other malicious parties could view in clear the private data used to train these neural networks.

With the advent of TEEs, especially AMD SEV SNP, computation can be done securely and confidentially, and once data is securely sent to the trainer, training happens in an isolated environment where not even very priveleged software like the operating system or the hypervisor could tamper.

BastionAI seeks, at its core, to protect user data by running executions in isolated, secure virtual machine where security guarantees are provided by the hardware.

<p class="comparison">
<a href="https://github.com/mithril-security/bastionai.git" class="bastionai">BastionAI🚀🔒</a> versus Federated Learning (FL).
</p>
Google's solution to distributed and secure training was federated learning. It's the scheme where model weights are distributed to all interested parties, trained with each party's private dataset and then the results aggregated in a distributed fashion. 

Although this solution protects user's private data to some extent, it is rid of several technical issues such as extreme communication costs, inavailability of enormous compute power for case where huge neural network models are trained, and very recently, data stealing through model weights. 

With BastionAI, once user's private data is sent securely sent through a trusted channel to the TEE, the model is trained in confidentiality and with differential privacy, data owner can express in quantities how much exposure they would want their data to be to models under training.

BastionAI solves the high cost of communication by centralizing computation (both training and testing of models,) and also increase confidentiality and integrity by using TEEs.

<p class="comparison">
<a href="https://github.com/mithril-security/bastionai.git" class="bastionai">BastionAI🚀🔒</a> versus Secure Multi-Party Computing.
</p>
BastionAI seeks to provide a new way of performing multi-party training but with less cryptography constraints and relinquishing trust to the TEE. Instead of using cumbersome mathematical facilities, we provide a centralized hardware-hardened environment were both model and data owners, distrusting one another, can successfully collaborate and train deep learning models.

Multi-party computation involves intense encryption through multiple cryptographic schemes and in a certain like training of models, this could be costly: both _communication cost_ and _computation cost_. Having a fortified centralized server responsible for training release the trainer and other interested parties from this cryptographic cost. 

## Telemetry
BastionAI collects anonymous data regarding general usage, this allows us to understand how you are using the project. We only collect data regarding the which models and datasets are being used, and the usage metrics.

This feature can be easily disabled, by settin up the environment variable `BASTIONAI_DISABLE_TELEMETRY` to 1.

## Contributing
There are many ways to contribute to the project:

- If you have any feedback on the project, share it under the `#bastionai-contributors` channel in the [Mithril Security Community Discord](https://discord.gg/TxEHagpWd4).
- Report issues you're facing and "Thumbs up" on issues and feature requests that are relevant to you.
- Investigate bugs and reviewing other developer's pull requests.
- Contributing code or documentation to the project by submitting a GitHub pull request. Check out the [Development Guide](CONTRIBUTING.md).

## Contact Us
### Ask a question
You can get community support in several ways:

Join our Discord Channel and engage in discussions with other users and the BastionAI and Mithril Security communities .
Ask a question about BastionAI and get community support by posting to [contact@mithrilsecurity.io](mailto:contact@mithrilsecurity.io). Posts can receive responses from community, and from engineers on the Mithril Security team who monitor the tag and offer unofficial support.

### File an issue or feature request

If you’re experiencing an issue, find out if there’s already a solution:

1. [Read the FAQs](FAQs.md)
2. Search existing issues in GitHub

If you still have an issue, file a new issue at mithril-security.

If you need help with a bug or error, include the following information in your message:

- What happened?
- What did you expect to happen?
- How can we reproduce the problem?

If you have an idea for a <a href="https://github.com/mithril-security/bastionai.git" class="bastionai">BastionAI🚀🔒</a> feature that isn’t already on the issue list of the project's GitHub, we’d like to hear from you! Please file a new feature request to the issue list so community members and Mithril Security engineers can review it.