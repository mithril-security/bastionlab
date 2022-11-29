## Benchmarks:

Benchmarking a 10Mx7 JOIN 100x5

|                        |         |                |                          | Timing             |                    |                                  |            |                              |
| ---------------------- | ------- | -------------- | ------------------------ | ------------------ | ------------------ | -------------------------------- | ---------- | ---------------------------- |
|                        | Privacy | Processor      | Memory Usage (MB)        | Standard deviation | Mean               | Dataset size                     | Operation  | Total Runs (Same Parameters) |
| BastionLab + TEE       | SEV     | AMD EPYC 7763v |                          | 0.1595178631       | 2.29914955550002 s | 10M x 7 (LHS) JOIN 100 x 5 (RHS) | INNER JOIN | 10                           |
| BastionLab             |         | AMD EPYC 7763v | (server) 12068.934364319 | 0.06813258905      | 1.64691811350021 s | 10M x 7 (LHS) JOIN 100 x 5 (RHS) | INNER JOIN | 10                           |
| Polars Lazy API Python |         | AMD EPYC 7763v | 16514.42188              | 0.05930174062      | 1.97007920599935 s | 10M x 7 (LHS) JOIN 100 x 5 (RHS) | INNER JOIN | 10                           |
| Pandas                 |         | AMD EPYC 7763v | 18422.98438              | 0.5469370083       | 11.7801000856994 s | 10M x 7 (LHS) JOIN 100 x 5 (RHS) | INNER JOIN | 10                           |
| Polars Lazy API Rust   |         | AMD EPYC 7763v | 11267.53197              | 0.05090933117      | 1.884 s            | 10M x 7 (LHS) JOIN 100 x 5 (RHS) | INNER JOIN | 10                           |


![](../../../assets/benchmark_amd_epyc_7763.png)

Benchmarking a 100Mx7 JOIN 100x5

|                        |         |                |                          | Timing             |                          |                                   |            |                              |       |        |
| ---------------------- | ------- | -------------- | ------------------------ | ------------------ | ------------------------ | --------------------------------- | ---------- | ---------------------------- | ----- | ------ |
|                        | Privacy | Processor      | Memory Usage (MB)        | Standard deviation | Mean                     | Dataset size                      | Operation  | Total Runs (Same Parameters) | Cores | Memory |
| BastionLab + TEE       | SEV     | AMD EPYC 7763v |                          | 0.07252514429      | 2.49139318740003 s       | 100M x 7 (LHS) JOIN 100 x 5 (RHS) | INNER JOIN | 10                           | 16    | 64 GB  |
| BastionLab             |         | AMD EPYC 7763v | (server) 22275.938194275 | 0.05010159207      | 1.99522447080016719000 s | 100M x 7 (LHS) JOIN 100 x 5 (RHS) | INNER JOIN | 10                           | 16    | 64 GB  |
| Polars Lazy API Python |         | AMD EPYC 7763v | 18120.21875              | 0.1378750987       | 2.433741233200635 s      | 100M x 7 (LHS) JOIN 100 x 5 (RHS) | INNER JOIN | 10                           | 16    | 64 GB  |
| Pandas                 |         | AMD EPYC 7763v | 18791.93359              | 0.6559093888       | 24.9457095852005 s       | 100M x 7 (LHS) JOIN 100 x 5 (RHS) | INNER JOIN | 10                           | 16    | 64 GB  |
| Polars Lazy API Rust   |         | AMD EPYC 7763v | 14339.72344              | 0.0521602339       | 4.507 s                  | 100M x 7 (LHS) JOIN 100 x 5 (RHS) | INNER JOIN | 10                           | 16    | 64 GB  |


![](../../../assets/benchmark_amd_epyc_7763_2.png)

Based on the above benchmarks we see that BastionLab performs operations faster than available solutions. There is a slight overhead when using BastionLab within a TEE but it is still as fast as Polars and much faster than Pandas. The overhead is due to the fact that the data is encrypted before it is sent to the TEE using AES-128. This is a one-time overhead and does not affect the performance of the operation itself. The memory usage is also quite low as shown by the previous tables. 