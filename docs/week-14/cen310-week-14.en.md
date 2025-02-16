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
header: 'CEN310 Parallel Programming Week-14'
footer: '![height:50px](http://erdogan.edu.tr/Images/Uploads/MyContents/L_379-20170718142719217230.jpg) RTEU CEN310 Week-14'
title: "CEN310 Parallel Programming Week-14"
author: "Author: Dr. Uğur CORUH"
date:
subtitle: "Quiz-2"
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

## Week-14 (Quiz-2)

#### Spring Semester, 2024-2025

---

## Quiz-2 Information

### Date and Time
- **Date:** May 16, 2025
- **Time:** 09:00-12:00 (3 hours)
- **Location:** Regular classroom

### Format
- Written examination
- Mix of theoretical and practical questions
- Both closed and open-ended questions

---

## Topics Covered

### 1. GPU Programming
- CUDA Architecture
- Memory Hierarchy
- Thread Organization
- Performance Optimization

### 2. Advanced Parallel Patterns
- Pipeline Processing
- Task Parallelism
- Data Parallelism
- Hybrid Approaches

### 3. Real-world Applications
- Scientific Computing
- Data Processing
- Matrix Operations
- N-body Simulations

---

## Sample Questions

### Theoretical Questions
1. Explain CUDA memory hierarchy and its impact on performance.
2. Compare different parallel patterns and their use cases.
3. Describe optimization strategies for GPU programs.

### Practical Problems
```cpp
// Question 1: What is the output of this CUDA program?
__global__ void kernel(int* data) {
    int idx = threadIdx.x;
    __shared__ int shared_data[256];
    
    shared_data[idx] = data[idx];
    __syncthreads();
    
    if(idx < 128) {
        shared_data[idx] += shared_data[idx + 128];
    }
    __syncthreads();
    
    if(idx == 0) {
        data[0] = shared_data[0];
    }
}

int main() {
    int* data;
    // ... initialization code ...
    kernel<<<1, 256>>>(data);
    // ... cleanup code ...
}
```

---

## Preparation Guidelines

### 1. Review Materials
- Lecture slides and notes
- Lab exercises
- Sample codes
- Practice problems

### 2. Focus Areas
- CUDA Programming
- Memory Management
- Performance Optimization
- Real-world Applications

### 3. Practice Exercises
- Write and analyze CUDA programs
- Implement parallel patterns
- Optimize existing code
- Measure performance

---

## Quiz Rules

1. **Materials Allowed**
   - No books or notes
   - No electronic devices
   - Clean paper for scratch work

2. **Time Management**
   - Read all questions carefully
   - Plan your time for each section
   - Leave time for review

3. **Answering Questions**
   - Show all your work
   - Explain your reasoning
   - Write clearly and organize your answers

---

## Grading Criteria

### Distribution
- Theoretical Questions: 40%
- Practical Problems: 60%

### Evaluation
- Understanding of concepts
- Problem-solving approach
- Code analysis and writing
- Performance considerations
- Clear explanations

---

## Additional Resources

### Review Materials
- CUDA Programming Guide
- Performance Optimization Guide
- Sample Applications
- Online Documentation:
  - [CUDA Documentation](https://docs.nvidia.com/cuda/)
  - [OpenMP Reference](https://www.openmp.org/)
  - [MPI Documentation](https://www.open-mpi.org/)

### Sample Code Repository
- Course GitHub repository
- Example implementations
- Performance benchmarks

---

## Contact Information

For any questions about the quiz:

- **Email:** ugur.coruh@erdogan.edu.tr
- **Office Hours:** By appointment
- **Location:** Engineering Faculty

---

<!-- _backgroundColor: aquq -->

<!-- _color: orange -->

# Good Luck!

--- 