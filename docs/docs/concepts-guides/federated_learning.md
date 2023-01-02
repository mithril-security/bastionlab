# Federated Learning
__________________________________________________________

In this guide we'll introduce Federated Learning, a machine learning technique that has been gaining a lot of popularity in the field of privacy-preserving data science. It's very relevant to us, because BastionLab's flow and architecture are inspired from it even if they diverge in key elements. 

We'll explain what Federated Learning is, why it is very promising and cover its current limitations. 

But first, let's take a quick detour to talk about the classical approach in the field right now. We'll see why we need to move from it in order to continue making progress while preserving data-privacy.

## Centralized data training
_______________________________________________________________

!!! quote "Definition"

	<font size="3"><i>Classical machine learning involves centralized data training, where the data is gathered, and the entire training process executes at the central server. Despite significant convergence, this training involves several privacy threats on participants' data when shared with the central cloud server. </i></font> [^1]

[^1]: [Federated Learning Versus Classical Machine Learning: A Convergence Comparison](https://arxiv.org/pdf/2107.10976)

What this definition means is that classical machine learning methods train data in a centralised 



___
One very popular emerging theory to solve those problems is ferederated learning. And it's great. The problem is that its success realies on technologies and ressources we don't really have at the moment (homomorphic encryption) or that are too expensive to be realistically adopted by industries.
Our solution comes up with a Federated Learning inspired approach, that would be more accessible. We call it Fortified Learning. 