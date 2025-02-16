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
header: 'CEN310 Parallel Programming Week-15'
footer: '![height:50px](http://erdogan.edu.tr/Images/Uploads/MyContents/L_379-20170718142719217230.jpg) RTEU CEN310 Week-15'
title: "CEN310 Parallel Programming Week-15"
author: "Author: Dr. Uğur CORUH"
date:
subtitle: "Final Project Review"
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

## Week-15 (Final Project Review)

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

## Final Project Requirements

### 1. Project Documentation
- Comprehensive project report
- Source code documentation
- Performance analysis results
- Implementation details
- Future work proposals

### 2. Technical Implementation
- Working parallel application
- Multiple parallel programming models
- Advanced optimization techniques
- Error handling and robustness
- Code quality and organization

---

## Presentation Guidelines

### Format
- 30 minutes per group
- 20 minutes presentation
- 10 minutes Q&A

### Content
1. Project Overview
   - Problem statement
   - Solution approach
   - Technical challenges

2. Implementation Details
   - Architecture design
   - Parallel strategies
   - Optimization techniques

3. Results and Analysis
   - Performance measurements
   - Scalability tests
   - Comparative analysis

4. Live Demo
   - System setup
   - Feature demonstration
   - Performance showcase

---

## Performance Analysis Requirements

### Metrics to Cover
- Execution time
- Speedup
- Efficiency
- Resource utilization
- Scalability

### Analysis Tools
```bash
# Performance measurement examples
$ nvprof ./cuda_program
$ mpirun -np 4 ./mpi_program
$ perf stat ./openmp_program
```

---

## Project Structure Example

```cpp
project/
├── src/
│   ├── main.cpp
│   ├── cuda/
│   │   ├── kernel.cu
│   │   └── gpu_utils.cuh
│   ├── mpi/
│   │   ├── communicator.cpp
│   │   └── data_transfer.h
│   └── openmp/
│       ├── parallel_loops.cpp
│       └── thread_utils.h
├── include/
│   ├── common.h
│   └── config.h
├── test/
│   ├── unit_tests.cpp
│   └── performance_tests.cpp
├── docs/
│   ├── report.pdf
│   └── presentation.pptx
├── data/
│   ├── input/
│   └── output/
├── scripts/
│   ├── build.sh
│   └── run_tests.sh
├── CMakeLists.txt
└── README.md
```

---

## Evaluation Criteria

### Technical Aspects (50%)
- Implementation quality (15%)
- Performance optimization (15%)
- Code organization (10%)
- Error handling (10%)

### Documentation (25%)
- Project report (10%)
- Code documentation (10%)
- Presentation quality (5%)

### Results & Analysis (25%)
- Performance results (10%)
- Comparative analysis (10%)
- Future improvements (5%)

---

## Common Project Topics

1. Scientific Computing
   - N-body simulations
   - Fluid dynamics
   - Monte Carlo methods
   - Matrix computations

2. Data Processing
   - Image/video processing
   - Signal processing
   - Data mining
   - Pattern recognition

3. Machine Learning
   - Neural network training
   - Parallel model inference
   - Data preprocessing
   - Feature extraction

4. Graph Processing
   - Path finding
   - Graph analytics
   - Network analysis
   - Tree algorithms

---

## Resources & References

### Documentation
- CUDA Programming Guide
- OpenMP API Specification
- MPI Standard Documentation
- Performance Optimization Guides

### Tools
- Visual Studio
- NVIDIA NSight
- Intel VTune
- Performance Profilers

---

## Project Report Template

### 1. Introduction
- Background
- Objectives
- Scope

### 2. Design
- System architecture
- Component design
- Parallel strategies

### 3. Implementation
- Development environment
- Technical details
- Optimization techniques

### 4. Results
- Performance measurements
- Analysis
- Comparisons

### 5. Conclusion
- Achievements
- Challenges
- Future work

---

## Contact Information

For project-related queries:

- **Email:** ugur.coruh@erdogan.edu.tr
- **Office Hours:** By appointment
- **Location:** Engineering Faculty

---

<!-- _backgroundColor: aquq -->

<!-- _color: orange -->

# Questions & Discussion

--- 