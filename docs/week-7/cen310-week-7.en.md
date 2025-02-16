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
header: 'CEN310 Parallel Programming Week-7'
footer: '![height:50px](http://erdogan.edu.tr/Images/Uploads/MyContents/L_379-20170718142719217230.jpg) RTEU CEN310 Week-7'
title: "CEN310 Parallel Programming Week-7"
author: "Author: Dr. Uğur CORUH"
date:
subtitle: "Quiz-1"
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

## Week-7 (Quiz-1)

#### Spring Semester, 2024-2025

---

## Quiz-1 Information

### Date and Time
- **Date:** March 28, 2025
- **Time:** 09:00-12:00 (3 hours)
- **Location:** Regular classroom

### Format
- Written examination
- Mix of theoretical questions and practical problems
- Both closed and open-ended questions

---

## Topics Covered

### 1. Introduction to Parallel Computing
- Parallel computing concepts
- Types of parallelism
- Performance metrics
- Architecture overview

### 2. OpenMP Programming
- Shared memory programming
- Thread management
- Data parallelism
- Synchronization
- Performance optimization

### 3. MPI Programming
- Distributed memory concepts
- Point-to-point communication
- Collective operations
- Common parallel patterns

---

## Sample Questions

### Theoretical Questions
1. Explain the difference between task parallelism and data parallelism.
2. What are the main considerations for load balancing in parallel programs?
3. Compare and contrast OpenMP and MPI programming models.

### Practical Problems
```cpp
// Question 1: What is the output of this OpenMP program?
#include <omp.h>
#include <stdio.h>

int main() {
    int x = 0;
    #pragma omp parallel num_threads(4) shared(x)
    {
        #pragma omp critical
        x++;
        #pragma omp barrier
        if (omp_get_thread_num() == 0)
            printf("x = %d\n", x);
    }
    return 0;
}
```

---

## Preparation Guidelines

### 1. Review Materials
- Course slides and notes
- Lab exercises
- Practice problems
- Sample codes

### 2. Focus Areas
- OpenMP directives and clauses
- MPI communication patterns
- Performance optimization techniques
- Parallel algorithm design

### 3. Practice Exercises
- Solve previous examples
- Write and analyze parallel programs
- Debug common issues
- Measure performance improvements

---

## Quiz Rules

1. **Materials Allowed**
   - No books or notes allowed
   - No electronic devices
   - Clean paper provided for scratch work

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
- Previous lecture slides
- Lab exercise solutions
- Practice problem sets
- Online documentation:
  - OpenMP: [https://www.openmp.org/](https://www.openmp.org/)
  - MPI: [https://www.open-mpi.org/](https://www.open-mpi.org/)

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