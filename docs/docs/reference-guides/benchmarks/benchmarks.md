# Benchmarks:
_________________________________

We don't want to improve data science privacy at a cost to your performance. This is why BastionLab uses Polars because it's built with Rust and is far more efficient than Pandas. 


BastionLab has several features that protect the data while still permitting operations to be performed on them.
Some of these include: authentication of users, access-control, logging, blocking or cancellation of operations if they do not respect the policy set by the data-owner.

BastionLab also supports execution within trusted execution environments (AMD SEV-SNP). This is useful in scenarios where the infrastructure that hosts the BastionLab server is untrusted. When run within a TEE, the infrastructure owner and others with access to the infrastructure will be unable to snoop on the data within BastionLab, even if they were to dump the memory as the memory is always encrypted.

Even with these privacy and security features, BastionLab is very fast as you’ll see in the benchmarks below.

They compare the following products:
BastionLab, BastionLab within a TEE, Polars Rust (using the Lazy API), Polars Python (which is the same as Polars Rust but has python bindings; also using the Lazy API), and Pandas.

All of the benchmarks use the same processor: AMD EPYC 7763v (with SEV-SNP disabled, except for BastionLab within a TEE which has SEV-SNP enabled).

_________________________________________________

### Benchmarking a 10Mx7 JOIN 100x5

This benchmark shows that BastionLab can perform operations faster than the other compared solutions - Pandas and Polars, two of the most popular data science libraries used today. 
The mean execution times show that BastionLab is comparable to Polars with BastionLab being 1.14 times faster than Polars Rust, and 1.19 times faster than Polars Python.

BastionLab is 7.15 times faster than Pandas.

Using BastionLab within a TEE adds a slight overhead. BastionLab (without a TEE) is 1.39 times faster than BastionLab within a TEE.


|                        | Privacy                           | Processor      | Memory Usage (MB) | Standard deviation of Time | Mean Execution Time | Operation  | Total Runs (Same Parameters) | Cores | Memory |
| ---------------------- | --------------------------------- | -------------- | ----------------- | -------------------------- | ------------------- | ---------- | ---------------------------- | ----- | ------ |
| BastionLab + TEE       | SEV (Encryption) + Access Control | AMD EPYC 7763v |                   | 0.15951                    | 2.29914 s           | INNER JOIN | 10                           | 16    | 64 GB  |
| BastionLab             | Access Control                    | AMD EPYC 7763v | 12068.93436       | 0.06813                    | 1.64691 s           | INNER JOIN | 10                           | 16    | 64 GB  |
| Polars Lazy API Python |                                   | AMD EPYC 7763v | 16514.42188       | 0.05930                    | 1.97007 s           | INNER JOIN | 10                           | 16    | 64 GB  |
| Pandas                 |                                   | AMD EPYC 7763v | 18422.98438       | 0.54693                    | 11.78010 s          | INNER JOIN | 10                           | 16    | 64 GB  |
| Polars Lazy API Rust   |                                   | AMD EPYC 7763v | 11267.53197       | 0.05090                    | 1.884 s             | INNER JOIN | 10                           | 16    | 64 GB  |


![](../../../assets/benchmark_amd_epyc_7763.png)

### Benchmarking a 100Mx7 JOIN 100x5

When a larger join operation is performed, we see that BastionLab remains comparable to Polars in terms of average execution time with BastionLab being 2.25 times faster than Polars Rust and 1.22 times faster than Polars Python.

BastionLab is 12.5 times faster than Pandas.

Running BastionLab within a TEE still presents the same overhead, with BastionLab (without a TEE) being 1.24 times faster than BastionLab within a TEE.


|                        | Privacy                           | Processor      | Memory Usage (MB) | Standard deviation of Time | Mean Execution Time | Operation  | Total Runs (Same Parameters) | Cores | Memory |
| ---------------------- | --------------------------------- | -------------- | ----------------- | -------------------------- | ------------------- | ---------- | ---------------------------- | ----- | ------ |
| BastionLab + TEE       | SEV (Encryption) + Access Control | AMD EPYC 7763v |                   | 0.07252                    | 2.49139 s           | INNER JOIN | 10                           | 16    | 64 GB  |
| BastionLab             | Access Control                    | AMD EPYC 7763v | 22275.93819       | 0.05010                    | 1.99522 s           | INNER JOIN | 10                           | 16    | 64 GB  |
| Polars Lazy API Python |                                   | AMD EPYC 7763v | 18120.21875       | 0.13787                    | 2.43374 s           | INNER JOIN | 10                           | 16    | 64 GB  |
| Pandas                 |                                   | AMD EPYC 7763v | 18791.93359       | 0.65590                    | 24.94570 s          | INNER JOIN | 10                           | 16    | 64 GB  |
| Polars Lazy API Rust   |                                   | AMD EPYC 7763v | 14339.72344       | 0.05216                    | 4.507 s             | INNER JOIN | 10                           | 16    | 64 GB  |


![](../../../assets/benchmark_amd_epyc_7763_2.png)

The memory benchmarks (memory usage) were tracked differently across rust applications and python application. In Rust we used jemalloc to track memory usage and memory_profiler in Python. 

When comparing memory usage benchmarks, we recommend comparing (python) Polars Python against Pandas and (rust) BastionLab against Polars Rust.

As seen in the benchmarks above, BastionLab performs operations faster than available solutions. There is a slight overhead when using BastionLab within a TEE but it is still as fast as Polars and significantly faster than Pandas.

These benchmarks were performed on Azure virtual machines, the specifications of the machines (cores and memory) can be found in each benchmark table.
