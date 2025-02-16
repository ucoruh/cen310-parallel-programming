---
marp: true
theme: default
style: |
    img[alt~="center"] {
      display: block;
      margin: 0 auto;
    }
_class: lead
paginate: true
backgroundColor: #fff
backgroundImage: url('https://marp.app/assets/hero-background.svg')
header: 'CEN310 Parallel Programming Week-8'
footer: '![height:50px](http://erdogan.edu.tr/Images/Uploads/MyContents/L_379-20170718142719217230.jpg) RTEU CEN310 Week-8'
title: "CEN310 Parallel Programming Week-8"
author: "Author: Dr. Uğur CORUH"
date:
subtitle: "Midterm Project Review"
geometry: "left=2.54cm,right=2.54cm,top=1.91cm,bottom=1.91cm"
titlepage: true
titlepage-color: "FFFFFF"
titlepage-text-color: "000000"
titlepage-rule-color: "CCCCCC"
titlepage-rule-height: 4
logo: "http://erdogan.edu.tr/Images/Uploads/MyContents/L_379-20170718142719217230.jpg"
logo-width: 100 
page-background:
page-background-opacity:
links-as-notes: true
lot: true
lof: true
listings-disable-line-numbers: true
listings-no-page-break: false
disable-header-and-footer: false
header-left:
header-center:
header-right:
footer-left: "© Dr. Uğur CORUH"
footer-center: "License: CC BY-NC-ND 4.0"
footer-right:
subparagraph: true
lang: en-US
math: katex
---

<!-- _backgroundColor: aquq -->

<!-- _color: orange -->

<!-- paginate: false -->

# CEN310 Parallel Programming

## Week-8 (Midterm Project Review)

#### Spring Semester, 2024-2025

---

## Project Review Day Schedule

### Morning Session (09:00-12:00)
- Project presentations (Group 1-4)
- Performance analysis discussions
- Q&A sessions

### Lunch Break (12:00-13:00)

### Afternoon Session (13:00-17:00)
- Project presentations (Group 5-8)
- Technical demonstrations
- Final feedback

---

## Project Requirements

### 1. Documentation
- Project report
- Source code documentation
- Performance analysis results
- Implementation challenges
- Future improvements

### 2. Implementation
- Working parallel program
- Use of OpenMP and/or MPI
- Performance optimizations
- Error handling
- Code quality

---

## Presentation Guidelines

### Format
- 20 minutes per group
- 15 minutes presentation
- 5 minutes Q&A

### Content
1. Problem Description
2. Solution Approach
3. Implementation Details
4. Performance Results
5. Challenges & Solutions
6. Demo

---

## Performance Analysis Requirements

### Metrics to Cover
- Execution time
- Speedup
- Efficiency
- Scalability
- Resource utilization

### Analysis Tools
```bash
# Example performance measurement
$ perf stat ./parallel_program
$ nvprof ./cuda_program
$ vtune ./openmp_program
```

---

## Example Project Structure

```cpp
// Project architecture example
project/
├── src/
│   ├── main.cpp
│   ├── parallel_impl.cpp
│   └── utils.cpp
├── include/
│   ├── parallel_impl.h
│   └── utils.h
├── tests/
│   └── test_parallel.cpp
├── docs/
│   ├── report.pdf
│   └── presentation.pptx
└── README.md
```

---

## Performance Comparison Template

### Sequential vs Parallel Implementation

```cpp
// Sequential implementation
double sequential_time = 0.0;
{
    auto start = std::chrono::high_resolution_clock::now();
    sequential_result = sequential_compute();
    auto end = std::chrono::high_resolution_clock::now();
    sequential_time = std::chrono::duration<double>(end-start).count();
}

// Parallel implementation
double parallel_time = 0.0;
{
    auto start = std::chrono::high_resolution_clock::now();
    parallel_result = parallel_compute();
    auto end = std::chrono::high_resolution_clock::now();
    parallel_time = std::chrono::duration<double>(end-start).count();
}

// Calculate speedup
double speedup = sequential_time / parallel_time;
```

---

## Common Project Topics

1. Matrix Operations
   - Matrix multiplication
   - Matrix decomposition
   - Linear equation solving

2. Scientific Computing
   - N-body simulation
   - Wave equation solver
   - Monte Carlo methods

3. Data Processing
   - Image processing
   - Signal processing
   - Data mining

4. Graph Algorithms
   - Shortest path
   - Graph coloring
   - Maximum flow

---

## Evaluation Criteria

### Technical Aspects (60%)
- Correct implementation (20%)
- Performance optimization (20%)
- Code quality (10%)
- Documentation (10%)

### Presentation (40%)
- Clear explanation (15%)
- Demo quality (15%)
- Q&A handling (10%)

---

## Project Report Template

### 1. Introduction
- Problem statement
- Objectives
- Background

### 2. Design
- Architecture
- Algorithms
- Parallelization strategy

### 3. Implementation
- Technologies used
- Code structure
- Key components

### 4. Results
- Performance measurements
- Analysis
- Comparisons

### 5. Conclusion
- Achievements
- Challenges
- Future work

---

## Resources & References

### Documentation
- OpenMP: [https://www.openmp.org/](https://www.openmp.org/)
- MPI: [https://www.open-mpi.org/](https://www.open-mpi.org/)
- CUDA: [https://docs.nvidia.com/cuda/](https://docs.nvidia.com/cuda/)

### Tools
- Performance analysis tools
- Debugging tools
- Profiling tools

---

<!-- _backgroundColor: aquq -->

<!-- _color: orange -->

# Questions & Discussion

--- 