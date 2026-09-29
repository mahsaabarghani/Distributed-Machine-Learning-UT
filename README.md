# Distributed Machine Learning

Coursework repository for the **Distributed Machine Learning** course at the **University of Tehran**, **Fall 2024**. The repository contains four computer assignments covering distributed numerical computation, CUDA programming, PyTorch distributed training, Apache Spark, Hugging Face Accelerate, Slurm, and PyTorch Profiler.

> **Repository description:** Course assignments from the University of Tehran Distributed Machine Learning course (Fall 2024), covering MPI, CUDA, PyTorch DDP, Spark, Slurm, Accelerate, mixed precision, and performance profiling.

![Python](https://img.shields.io/badge/Language-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch%20Distributed-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/Compute-CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white)
![MPI](https://img.shields.io/badge/Communication-MPI-4B5563?style=flat-square)
![Spark](https://img.shields.io/badge/Engine-Apache%20Spark-E25A1C?style=flat-square&logo=apachespark&logoColor=white)
![Slurm](https://img.shields.io/badge/Cluster-Slurm-2563EB?style=flat-square)
![Accelerate](https://img.shields.io/badge/Library-Hugging%20Face%20Accelerate-FFD21E?style=flat-square)
![Distributed Training](https://img.shields.io/badge/Topic-Distributed%20Training-7C3AED?style=flat-square)
![Profiler](https://img.shields.io/badge/Tool-PyTorch%20Profiler-0EA5E9?style=flat-square)
![Evaluation](https://img.shields.io/badge/Focus-Performance%20Evaluation-F59E0B?style=flat-square)
![University](https://img.shields.io/badge/University-Tehran-1D4ED8?style=flat-square)
![Semester](https://img.shields.io/badge/Semester-Fall%202024-475569?style=flat-square)
![Coursework](https://img.shields.io/badge/Type-Academic%20Coursework-7C3AED?style=flat-square)

> **University:** University of Tehran  \
> **Course:** Distributed Machine Learning  \
> **Semester:** Fall 2024  \
> **Student:** Mahsa Abarghani  \
> **Student number:** 810103053

## Overview

These exercises study how machine-learning workloads behave when computation is distributed across CPU cores, GPUs, multiple machines, and Spark workers. The work combines implementation with experimental analysis of execution time, model accuracy, memory usage, communication backends, scaling behavior, and profiling results.

The repository has exactly four assignment folders. Each folder contains the relevant assignment description, source code, reports, datasets, scripts, notebooks, and supporting resources for that assignment.

## Learning objectives

The assignments build a progression from process-level parallelism to distributed deep-learning workflows:

- Implement distributed numerical algorithms with MPI and measure the effect of nodes and CPU cores.
- Compare serial, optimized, BLAS, and MKL implementations for computational workloads.
- Write CUDA kernels and compare CPU and GPU execution for image processing.
- Train neural networks with PyTorch multiprocessing and distributed data-parallel techniques.
- Study how batch size, GPU count, communication backend, and process placement affect runtime, memory, and accuracy.
- Use Apache Spark for distributed text processing, embeddings, and clustering.
- Run distributed training with `torchrun`, Slurm, and Hugging Face Accelerate.
- Evaluate mixed precision and use PyTorch Profiler to identify the cost of model modules.

## Experimental perspective

The submitted work records more than source code. Each assignment also contains reports, scripts, notebooks, datasets, and supporting media so that the experiments can be reviewed and reproduced in their original context. The main measurements discussed across the repository are:

| Measurement | Examples |
| --- | --- |
| Runtime | Single-node versus multi-node execution, CPU versus GPU execution, and different batch sizes. |
| Accuracy | Final test accuracy for distributed neural-network training. |
| Memory | GPU memory usage and module-level CPU memory measurements. |
| Scalability | Changes caused by process count, node count, and communication strategy. |
| Profiling | Time and memory used by `Linear`, `BatchNorm`, and activation modules. |

## Assignment map

| Folder | Main topics | Main source materials |
| --- | --- | --- |
| [`CA1/`](CA1/) | MPI, distributed numerical computation, logistic regression, and BLAS/MKL benchmarking | `DMLS CA1.zip`, `MahsaAbarghani-810103053.zip` |
| [`CA2/`](CA2/) | CUDA grayscale conversion, PyTorch multiprocessing, multi-GPU training, batch-size analysis, and Gloo/NCCL | `CA2_Q1.ipynb`, `810103053-tamrine2-mahsa-abarghani.rar`, `CUDAProgramming.zip` |
| [`CA3/`](CA3/) | Spark RDD, n-gram analysis, Word2Vec, and K-Means | `CA3.zip`, `810103053.rar` |
| [`CA4/`](CA4/) | `torchrun`, Slurm, Accelerate, mixed precision, and PyTorch Profiler | `810103053#4.zip`, `train_data.zip`, `test_data.zip` |

## CA1 - MPI and distributed computing

CA1 explores distributed numerical computation with MPI and cluster execution. It includes Taylor-series computation, work distribution across nodes and cores, distributed logistic regression, and supplementary BLAS/MKL matrix benchmarks.

- [Assignment description](CA1/ca1-assignment.pdf)
- [Question 1 - Taylor series](CA1/question-1-taylor-series/)
- [Question 2 - distributed logistic regression](CA1/question-2-distributed-logistic-regression/)
- [Supplementary matrix benchmarks](CA1/supplementary-matrix-benchmarks/)
- [MPI course hands-on resource](CA1/resources/mpi-course-hands-on.mp4)

The question folders preserve the submitted Python and shell scripts together with the corresponding reports and input arrays.

## CA2 - CUDA and PyTorch distributed training

CA2 uses the STL-10 image dataset to compare Python and CUDA image processing and to study distributed neural-network training. It covers four questions:

1. RGB-to-grayscale conversion with Python and a CUDA kernel.
2. Single-GPU and multi-GPU CNN training with PyTorch multiprocessing.
3. The effect of batch size on training time, accuracy, and GPU memory.
4. A comparison of the Gloo and NCCL communication backends.

The original `CA2_Q1.ipynb` notebook is kept at the beginning of this folder as the source notebook for Question 1:

- [CA2 Question 1 source notebook](CA2/CA2_Q1.ipynb)
- [Assignment description](CA2/ca2-assignment.pdf)
- [CA2 report](CA2/ca2-report.pdf)
- [Question 1 - CUDA grayscale conversion](CA2/question-1-cuda-grayscale/)
- [Question 2 - multi-GPU training](CA2/question-2-multi-gpu-training/)
- [Question 3 - batch-size analysis](CA2/question-3-batch-size-analysis/)
- [Question 4 - communication backends](CA2/question-4-communication-backends/)
- [CUDA programming resource](CA2/resources/cuda-programming.mp4)

## CA3 - Spark

CA3 uses Apache Spark for distributed text processing and machine-learning workflows. The submitted material includes RDD and n-gram analysis on the Shahnameh text, Word2Vec experiments, and customer clustering with K-Means.

- [Assignment description](CA3/ca3-assignment.pdf)
- [CA3 report](CA3/ca3-report.pdf)
- [Question 1 - RDD and n-gram analysis](CA3/question-1-rdd-and-ngram/)
- [Question 2 - Word2Vec](CA3/question-2-word2vec/)
- [Question 3 - K-Means](CA3/question-3-k-means/)
- [Spark/HDFS hands-on resource](CA3/resources/spark-hdfs-hands-on.mp4)
- [Shahnameh text](CA3/resources/shahname.txt)
- [Customer data](CA3/resources/customers.csv)

## CA4 - distributed training and profiling

CA4 focuses on distributed neural-network training and performance analysis. It uses the provided training and test arrays throughout the assignment. The four questions cover:

1. Distributed training with `torchrun` and Slurm.
2. Distributed training with Hugging Face Accelerate and Slurm.
3. Mixed-precision training and comparison of supported precision modes.
4. CPU profiling of `Linear`, `BatchNorm`, `ReLU`, `Sigmoid`, `Tanh`, and `GeLU` modules.

The `train_data` and `test_data` folders are intentionally inside CA4 because they are required by this assignment:

- [Assignment description](CA4/ca4-assignment.pdf)
- [CA4 report](CA4/ca4-report.pdf)
- [Training data](CA4/train_data/)
- [Test data](CA4/test_data/)
- [Question 1 - `torchrun` and Slurm](CA4/question-1-torchrun-slurm/)
- [Question 2 - Accelerate and Slurm](CA4/question-2-accelerate-slurm/)
- [Question 3 - mixed precision](CA4/question-3-mixed-precision/)
- [Question 4 - PyTorch Profiler](CA4/question-4-pytorch-profiler/)

## Tools and concepts

![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![PySpark](https://img.shields.io/badge/Framework-PySpark-E25A1C?style=flat-square&logo=apachespark&logoColor=white)
![Accelerate](https://img.shields.io/badge/Library-Hugging%20Face%20Accelerate-FFD21E?style=flat-square)
![Profiler](https://img.shields.io/badge/Tool-PyTorch%20Profiler-0EA5E9?style=flat-square)

- MPI and process-level communication with `mpi4py`
- CPU and GPU parallelism
- CUDA kernels and GPU memory behavior
- PyTorch multiprocessing and distributed training
- Gloo and NCCL communication backends
- Spark RDD and MLlib-style workflows
- Slurm-based cluster execution
- Mixed-precision training
- Runtime, accuracy, memory, and profiler analysis

## Repository structure

```text
Distributed-Machine-Learning/
├── CA1/
├── CA2/
├── CA3/
├── CA4/
└── README.md
```

## Notes

- All original files on `D:\` were preserved. This repository contains extracted and renamed copies.
- Duplicate copies of the CA2 assignment PDF were consolidated into one normalized file, `CA2/ca2-assignment.pdf`.
- The CA4 training and test datasets remain inside `CA4/` as requested.
- The CA2 CUDA video is approximately 103.5 MB. GitHub may require Git LFS for that file, or it can be kept outside the repository when publishing.
