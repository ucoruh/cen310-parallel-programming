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
header: 'CEN310 Parallel Programming Week-6'
footer: '![height:50px](http://erdogan.edu.tr/Images/Uploads/MyContents/L_379-20170718142719217230.jpg) RTEU CEN310 Week-6'
title: "CEN310 Parallel Programming Week-6"
author: "Author: Dr. Uğur CORUH"
date:
subtitle: "Performance Optimization"
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

## Week-6

#### Performance Optimization

---

## Outline

1. Performance Analysis Tools
   - Profilers
   - Hardware Counters
   - Performance Metrics
   - Bottleneck Detection
   - Benchmarking

2. Memory Optimization
   - Cache Optimization
   - Memory Access Patterns
   - Data Layout
   - False Sharing
   - Memory Bandwidth

3. Algorithm Optimization
   - Load Balancing
   - Work Distribution
   - Communication Patterns
   - Synchronization Overhead
   - Scalability Analysis

4. Advanced Optimization Techniques
   - Vectorization
   - Loop Optimization
   - Thread Affinity
   - Compiler Optimizations
   - Hardware-Specific Tuning

---

## 1. Performance Analysis Tools

### Using Profilers

Example using Intel VTune:
```cpp
#include <omp.h>
#include <vector>

void optimize_matrix_multiply(const std::vector<float>& A,
                            const std::vector<float>& B,
                            std::vector<float>& C,
                            int N) {
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            float sum = 0.0f;
            // Cache-friendly access pattern
            for(int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Performance measurement
void measure_performance() {
    const int N = 1024;
    std::vector<float> A(N * N), B(N * N), C(N * N);
    
    // Initialize matrices
    for(int i = 0; i < N * N; i++) {
        A[i] = rand() / (float)RAND_MAX;
        B[i] = rand() / (float)RAND_MAX;
    }
    
    double start = omp_get_wtime();
    optimize_matrix_multiply(A, B, C, N);
    double end = omp_get_wtime();
    
    printf("Time: %f seconds\n", end - start);
}
```

---

## 2. Memory Optimization

### Cache-Friendly Data Access

```cpp
// Bad: Cache-unfriendly access
void bad_access(float* matrix, int N) {
    for(int j = 0; j < N; j++) {
        for(int i = 0; i < N; i++) {
            matrix[i * N + j] = compute(i, j);
        }
    }
}

// Good: Cache-friendly access
void good_access(float* matrix, int N) {
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            matrix[i * N + j] = compute(i, j);
        }
    }
}
```

---

### False Sharing Prevention

```cpp
// Bad: False sharing
struct BadCounter {
    int count;  // Multiple threads updating adjacent memory
};

// Good: Padding to prevent false sharing
struct GoodCounter {
    int count;
    char padding[60];  // Align to cache line size
};

void parallel_count() {
    const int NUM_THREADS = 4;
    GoodCounter counters[NUM_THREADS];
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        for(int i = 0; i < 1000000; i++) {
            counters[tid].count++;
        }
    }
}
```

---

// ... continue with detailed content for Week-6 