
# **Week 1 — GPU Intuition & Compute Foundations**


##  **Learning Goals**

By the end of this week, you should be able to:

* Explain the architectural differences between CPUs and GPUs
* Understand SIMT parallelism, warps, and throughput computing
* Identify workloads that benefit from GPU acceleration
* Understand the CUDA execution hierarchy (threads → blocks → grids)
* Identify at least one potential GPU-accelerable task from your own research or interests

This week builds your mental model so everything later (memory, performance, kernels) fits together.


## **Concepts Covered This Week**

* CPU vs GPU architecture
* SIMD vs SIMT
* Warps, streaming multiprocessors, and massive parallelism
* Throughput computing vs latency computing
* CUDA execution hierarchy
* Identifying GPU-friendly workloads
* Profiling intuition: *what part of your code is slow? why?*


## **Required Resources**

### **1. GPU Architecture Fundamentals**

**UC Berkeley CS61C — GPU Architecture Lecture** (James Percy)  
[https://www.youtube.com/watch?v=xdcW52tEPfE](https://www.youtube.com/watch?v=xdcW52tEPfE)  
**Focus**: Why CPUs hit a wall, power/memory constraints, throughput vs latency philosophy

**NVIDIA CUDA Programming Guide — Chapter 1**  
[https://docs.nvidia.com/cuda/cuda-c-programming-guide/](https://docs.nvidia.com/cuda/cuda-programming-guide/) 
Read: Sections 1.1–1.3  
**Focus**: SIMT execution model, thread/block/grid hierarchy, host vs device


### **2. GPU in Practice**

**Stanford CS149 — GPU Architecture & Programming Model** (Kayvon Fatahalian)  
[https://gfxcourses.stanford.edu/cs149/fall24/lecture/gpuarch/](https://gfxcourses.stanford.edu/cs149/fall24/lecture/gpuarch/)  
Video: [https://www.youtube.com/watch?v=gNyEHuuFdPQ](https://www.youtube.com/watch?v=gNyEHuuFdPQ)  
**Focus**: Warps, SMs, memory hierarchy basics, what makes code "GPU-friendly"

**Mark Harris — "An Even Easier Introduction to CUDA"**  
[https://developer.nvidia.com/blog/even-easier-introduction-cuda/](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)  
Read: Full post (15 mins)  
**Focus**: First complete CUDA example walkthrough (you don't need to code yet)


## **Supplementary Resources** 

**For ML Practitioners:**
- PyTorch CUDA Semantics: [https://pytorch.org/docs/stable/notes/cuda.html](https://pytorch.org/docs/stable/notes/cuda.html) (10 min skim — understand `.cuda()` and `.to(device)`)

**For Systems/HPC Folks:**
- CUDA Refresher (Memory Hierarchy): [https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/)

**For Visual Thinkers:**
- GPU Gems Chapter 39 (Parallel Prefix Sum): [https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda) — See how algorithms change for GPUs

## **Week 1 Assignment (Summary)**

This week’s assignment focuses on **analyzing a workload** you may want to accelerate later in the course and understanding how it fits into the GPU execution model. You will not write CUDA code yet.

### **Task 1 — Identify & Analyze a GPU-Accelerable Workload**

Choose a computation-heavy task from your research or interests (e.g., a simulation loop, numerical kernel, ML operation, or data processing step).
Write a **technical analysis (1–2 pages)** covering:

* **Operation breakdown:** pseudocode, data shapes, parallelism opportunities
* **Compute vs memory behavior:** compute-bound vs memory-bound, data reuse, dependencies
* **Expected GPU mapping:** how threads, blocks, and grids would map to the iteration space, expected scaling behavior, potential issues (memory access, divergence)

**Note:** You do *not* need to provide a CPU runtime baseline this week.
You have **until Week 3** to submit `cpu_baseline.py`.

### **Task 2 — CUDA Execution Model Diagram**

Create a diagram illustrating:

```
Grid → Blocks → Warps → Threads
```

Add annotations showing:

* How your chosen workload maps to this hierarchy
* Where synchronization might be required
* Where memory bottlenecks may occur
* Where shared memory could be used (conceptually)

Upload as an image or PDF.


### **Submission Folder**

Place your files here:

```
week1/
 ├── assignment1-<name>.pdf
 └── cpu_baseline-<name>-<task>.py   # optional until Week 3
```


## **Optional Extra Resources**

* “Even Easier Introduction to CUDA” — NVIDIA Blog
  [https://developer.nvidia.com/blog/even-easier-introduction-cuda/](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
* “How GPUs Work” — Stanford CME193 notes

