
The method used here for testing was the same as in the previous benchmarking example with AMD EPYC 7763.

For these benchmarks, we set the batch size to be the maximum possible, fitting in 16GB of GPU RAM, using a NVIDIA V100 GPU.

![](../../../../assets/v100_exec_times_16GB.png)

By using 40GB of GPU RAM on a NVIDIA A100 GPU, you can obtain these results :

![](../../../../assets/a100_exec_times_40GB.png)

Here you can have a look at the precise execution times.


|                                   |  Privacy   | Nvidia V100 Training Time (s/epoch) | Nvidia V100 Batch size |
| --------------------------------- | :--------: | :---------------------------------: | :--------------------: |
| Pytorch (baseline)                |    None    |                7,87                 |        2 048,00        |
| PyTorch + Opacus                  | Incomplete |               197,50                |         32,00          |
| PyTorch + 2-party Flower          | Incomplete |                26,48                |        4 096,00        |
| Pytorch + 2-party Flower + Opacus |     OK     |               336,79                |         32,00          |
| BastionAI (no DP, no TEE)         | Incomplete |                11,81                |        4 096,00        |
| BastionAI (DP, no TEE)            | Incomplete |               126,15                |         256,00         |
| BastionAI (no DP, TEE)            | Incomplete |                 N/A                 |        4 096,00        |
| BastionAI (DP & TEE)              |     OK     |                 N/A                 |         256,00         |


|                                   |  Privacy   | Nvidia A100 Training Time (s/epoch) | Nvidia A100 Batch size |
| --------------------------------- | :--------: | :---------------------------------: | :--------------------: |
| Pytorch (baseline)                |    None    |                7,06                 |        2 048,00        |
| PyTorch + Opacus                  | Incomplete |                61,49                |         128,00         |
| PyTorch + 2-party Flower          | Incomplete |                24,47                |        4 096,00        |
| Pytorch + 2-party Flower + Opacus |     OK     |               101,19                |         128,00         |
| BastionAI (no DP, no TEE)         | Incomplete |                6,51                 |        4 096,00        |
| BastionAI (DP, no TEE)            | Incomplete |                82,60                |         256,00         |
| BastionAI (no DP, TEE)            | Incomplete |                10,70                |        4 096,00        |
| BastionAI (DP & TEE)              |     OK     |               105,71                |         256,00         |
