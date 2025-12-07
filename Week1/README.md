
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

### **1. NVIDIA CUDA Programming Guide (Conceptual Introduction)**

Read: **Chapter 1 — Introduction**
[https://docs.nvidia.com/cuda/cuda-programming-guide/](https://docs.nvidia.com/cuda/cuda-programming-guide/)

Focus on:

* What GPUs are optimized for
* Parallel execution model
* Overview of CUDA threads/blocks/grids



### **2. UC Berkeley CS61C — Parallel Processors (Lecture 17)**

Video:
[https://www.youtube.com/watch?v=gychBEOgG8A](https://www.youtube.com/watch?v=gychBEOgG8A)
(If link breaks, search: *CS61C Parallel Processors Lecture 17*)

This gives the “physics layer” of performance:

* Why CPUs stopped getting faster
* Why GPUs thrive on massive parallelism
* Memory bottlenecks and throughput considerations



### **3. Stanford CS231n — Hardware/Software Interface (Lecture 15)**

Slides & Video:
[https://www.youtube.com/watch?v=WGf1f2HbJpE](https://www.youtube.com/watch?v=WGf1f2HbJpE)

Focus on:

* GPU in the modern AI stack
* How frameworks like PyTorch map operations to hardware


### **Week 1 Assignment (Summary)**

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
 ├── assignment.pdf
 └── cpu_baseline.py   # optional until Week 3
```


## **Optional Extra Resources**

* “Even Easier Introduction to CUDA” — NVIDIA Blog
  [https://developer.nvidia.com/blog/even-easier-introduction-cuda/](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
* “How GPUs Work” — Stanford CME193 notes

