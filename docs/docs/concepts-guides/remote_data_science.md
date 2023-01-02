# Fortified data science
__________________________________________________________

How to keep data science collaboration interactive, efficient, private and secure, when working with remote third-parties? This question isn't easy to answer because ensuring one of those four elements often come at the cost of another with the present technology. As a result, collaborations are either dropped before they can ever happen - especially in fields dealing with very sensitive data like health, finance, advertising -, or they happen... with no proper privacy garantees.

In this guide we'll talk about fortified data science, a technique we came up with to solve this problem and offer an overall good experience across the board. It is the theory behind BastionLab flow and architecture, as well as its main API object: the RemoteLazyFrame. 

But before diving into the specifics of fortified data science, let's talk about the current solutions and why they pose many threats to data privacy. 

## Current methods and their risks
_______________________________________________________________

### Opening remote access to a Jupyter notebook

In this first case, the data owner doesn't know much about data science in general. They just have a big quantity of data they want to explore and ask a data scientist to analyse it. To do so, they'll give the data scientist access to a remote jupyter notebook, that has access to the patient data. They'll ask the data scientist to NOT download the data on their computer, and tell them they are only allowed to use it via the Jupyter notebook.

This approach technically works - the data scientist can efficiently and easily do their job with it - but it relies entirely on trust and poses big privacy and security risks. The data scientist has **direct** access to the data. They can see it, download it, or do whatever they want with it. The only thing preventing them to do so is a promise that they won't. 

That's already a big problem, but there is worse: the data owner will never know that their data might have been tampered with. There is no native solution to track the operations done by the data scientist on their data, which means it is almost impossible for the data owner to prosecute anyone exploiting the data: they can’t even prove it's been misused.

### Sending Python scripts 

In this other case, the data owner has an IT department with expert python programmers. Instead of letting the data scientist access the data directly, they want to do everything through python scripts sent by email that they'll review and run themselves before sending the results. 

This approach is a good step forward in terms of security and privacy, but it operates at the cost of interactivity and efficiency. Actually, in many cases, it won't even be possible because the data owner needs to have an expert Python programmer to audit the data scientist's code before running it, which is very rare. 

But let's say the data owner has a Python expert: there are countless ways to hide malicious code in a python script, that even them won’t notice. For example, it is very easy to hide code that downloads the whole dataset! 

Finally, data science is an interactive process. You'll often need to try a quick piece of code, submit it, get the answer you need ; then try another quick piece of code, submit it, get the answer you need ; then repeat that process many, many times. This is the whole reason Jupyter notebooks exist in the first place and why the Python script method probably doesn't have a promising future.

### Database anonymization

In any case, data owners need to anonymize the data they'll give access to through a Jupyter notebook or execute Python scripts on. There are many techniques to do so (*data masking, pseudonymization, generalization, data swapping, data perturbation, synthetic data...*) but they often have to be done manually. Not only will the process lack efficiency, the risk of human error will always be looming over the results.

It is quite easy to deanonymise databases (*the reverse action which often consists of cross-referencing the informations from the data owner's dataset with public databases*), because informations like age or sex, for example, can be used to identify a person by name. To avoid this, techniques exist like showing only aggregated results or using differential privacy.

## 

## Why we need it
________________________________________________________________

Data owners often need or wish that remote data scientists would access their datasets - like a hospital might want to valorize their data to external parties, startups, labs, or receive help from external experts, for instance. 

But how can we open data to outsiders while giving a good user-experience for the data scientist and providing a high-enough level of security and privacy for the shared data? 

One of the most popular solutions is to give access to a Jupyter Python notebook installed on the data owner infrastructure. This is dangerous, because it exposes the dataset to serious data leakages. Jupyter was *not* made for this task and exfiltrating data can easily be done. For example, rows can be printed little by little until the whole dataset is extracted, or a malicious data scientist could exfiltrate a trained neural network in which the whole dataset has been hidden in the weights.

This is possible because the data scientist can run arbitrary Python scripts on the data to perform extraction attacks. It makes it extremely complicated to try to analyze their queries to detect fraudulent behavior: static and dynamic analysis tools would be inefficient because Python code is too dynamic and expressive.

That is why we have built BastionLab, a data science framework to perform remote and secure Exploratory Data Analysis. The following plan illustrates the architecture of our solution and the workflow we implement:

![](../../assets/BastionLab_Workflow.png)

There are a few key differences between a remotely accessed Jupyter notebook and the use of our remote data science framework.

In the remotely accessed Jupyter notebook, the data scientist has direct access to the data, can run arbitrary code on it, and can see the results of the output. 

In the remote data science framework, only a restricted set of operations is authorized on the data. Instead of allowing  arbitrary remote code execution, the only *execute operations* allowed are ones needed for data science, like joins, mean, train, etc. Those operations can be analyzed and blocked if they don't answer the data owner's privacy policy.

### Takeaways

When remotely connecting to a Jupyter notebook

-   The data scientist basically has a direct access to the entirety of the dataset

-   They can also send arbitrary Python code to be executed and exfiltrate data

A remote data science framework comes to solve that problem by ensuring the data scientist can *only* access the dataset and the database through a sanitized interface that allows the data owner to have full control.

## How it works
_______________________________________________________

A remote data science framework acts as a filter barrier. In the case of BastionLab, only pre-coded operations approved by the data owner can be run on the data. The results shared with the data scientist are also finely controlled. 

This is enabled by four principles:

-   The data scientist never has access to the data directly

-   The data scientist manipulates on their machine only a local object that contains metadata to interact with a remotely hosted dataset

-   The data scientist interacts with the remotely hosted objects only through remote proccedure calls that are transparent to them

-   Those requests are analyzed and matched against a privacy policy defined by the data owner, so that only allowed operations are executed and sanitized results returned

By incorporating those four principles, it becomes possible for a data scientist to interact with a data scientist remotely, and iterate quickly on it, without sacrificing security and privacy because:

Since remote data science faces terrible consequences in case of leaks, it's critical to ensure:

-   That no data is exposed in clear

-   That only a certain set of sanitized operations can be executed on the data

-   That the operations executed during a session are always matched against the privacy policy - to make sure that only allowed operations can be used on the data

## How we do it
__________________________________________________

In this part, we'll go over the details of how remote data science is implemented in BastionLab. 

We'll outline three different concepts and provide an in-depth presentation of their implementation through the RemoteLazyFrame object.

### Remote Objects

BastionLab works with Remote Objects to guarantee both privacy and ease-of-use by data scientists .

From the data scientist's perspective, Remote Objects are a pointer to a remote resource, hosted on the data owner infrastructure. They serve as an abstraction to query the remote dataset *as if* the data were available locally, but technically it is **never** the case.

Queries are often crafted in a lazy manner locally - this means that nothing is executed locally (which is logical as the data is *not* on the data scientist's perimeter). When we want to run a computation, then some methods can be used to serialize the request and send it for remote execution. 

Our solution, RemoteLazyFrames, is one method to do so. We'll introduce it in a few sections down this guide.

### Limited expressivity

Arbitrary code execution, as we saw previously, can be a cause of great headaches in terms of security. That is why we aim to reduce as much as possible the amount of code we need to trust (also called the Trusted Execution Base). 

To do so, we provide a computational engine specific to the task at hand, instead of providing a Python interpreter to the data scientist. This reduces drastically the ability of the data scientist to inject malicious code to extract the datasets. It also allows us to better optimize the computation executed because it fits a specific pattern (like neural network or data frame query).

### Sanitized outputs

To sanitize outputs, we allow a  limited number of operators to be executed by the data scientist. We also make sure that the only outputs which can be shared respect a privacy policy defined by the data owner.

For example, we could only allow differentially private outputs to be shared, or not allow raw information to be printed on the data scientist interface.

### Our Solution: the **RemoteLazyFrame**
___________________________________________________________

The RemoteLazyFrame, a remote privacy-friendly version of a DataFrame, implements all these features:

-   **Remote Object:** the RemoteLazyFrame is a pointer to a remote data frame. Queries are built locally in a lazy method, are sent for execution through the `.collect()` method, and results are pulled with `.fetch()`

-   **Limited expressivity:** only polars operators are supported, like mean, join, std, etc. This means no Python code is directly executed on the data.

-   **Sanitized outputs:** a privacy policy can be put in place to only allow aggregate results to be shared or if rows have to be shared, only after approval from the data owner.

BastionLab allows the data owner to specify the constraints under which data scientists may download results by customizing an access policy. Access policies are defined for each remote object sent by the data owner and are inherited by the results computed using these remote objects. 

They support two access modes, either automatic or by requesting the data owner's approval, that both correspond with a set of constraints. The checks run as follows: if the constraints for automatic access are fulfilled, the server directly returns the data. Otherwise, it checks the constraints for approval mode, and, if they are fulfilled, sends a request to the data owner who can accept or reject it. If none of the two sets of constraints are fulfilled, the data owner cannot download the results.

The default policy used for DataFrames automatically accepts that the data scientist downloads aggregated results (with a constraint on the number of rows of the original DataFrame per group) and requires the data owner's approval in all other cases. In addition to this mechanism, the data owner can define a blacklist of columns that would be wiped (replaced with '\*') if downloaded.
