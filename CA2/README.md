# Assignment 2 - CUDA and PyTorch DDP

This assignment studies CUDA image processing and distributed neural-network training on STL-10 data.

The main submission files are kept directly in this folder:

- [`CA2_Q1.ipynb`](CA2_Q1.ipynb)
- [`ca2-assignment.pdf`](ca2-assignment.pdf)
- [`ca2-report.pdf`](ca2-report.pdf)
- [`cuda-programming.mp4`](resources/cuda-programming.mp4)

## Questions

| Question | Folder | Main deliverable |
| --- | --- | --- |
| 1 | [`question-1-cuda-grayscale/`](question-1-cuda-grayscale/) | Python/CUDA RGB-to-grayscale notebooks |
| 2 | [`question-2-multi-gpu-training/`](question-2-multi-gpu-training/) | Single-GPU and multi-GPU classifiers |
| 3 | [`question-3-batch-size-analysis/`](question-3-batch-size-analysis/) | Batch-size experiment analysis |
| 4 | [`question-4-communication-backends/`](question-4-communication-backends/) | Gloo/NCCL comparison |

Questions 3 and 4 reuse the multi-GPU training implementation from Question 2. The train/test arrays used by the experiments are kept in [`../CA4/`](../CA4/).

- [Assignment description](ca2-assignment.pdf)
- [Assignment report](ca2-report.pdf)
