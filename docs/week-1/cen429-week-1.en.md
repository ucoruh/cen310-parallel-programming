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
header: 'CEN310 Parallel Programming Week-1'
footer: '![height:50px](http://erdogan.edu.tr/Images/Uploads/MyContents/L_379-20170718142719217230.jpg) RTEU CEN310 Week-1'
title: "CEN310 Parallel Programming Week-1"
author: "Author: Asst. Prof. Dr. Uğur CORUH"
date:
subtitle: "Course Introduction and Development Environment Setup"
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
footer-left: "© Dr. Ugur CORUH"
footer-center: "License: CC BY-NC-ND 4.0"
footer-right:
subparagraph: true
lang: en-US
math: katex
charset: "utf-8"
tags:
  - cen310-week-1
  - parallel-programming
  - course-introduction
  - development-environment
  - spring-2025
ref_link: na
---

<!-- _backgroundColor: aquq -->

<!-- _color: orange -->

<!-- paginate: false -->

# CEN310 Parallel Programming

## Week-1

#### Course Introduction and Development Environment Setup

---

Download 

- [PDF](pandoc_cen310-week-1.pdf)
- [DOC](pandoc_cen310-week-1.docx)
- [SLIDE](cen310-week-1.pdf)
- [PPTX](cen310-week-1.pptx)

---

<iframe width=700, height=500 frameBorder=0 src="../cen310-week-1.html"></iframe>

---

## Outline (1/3)

1. Course Overview
   - Course Description
   - Learning Outcomes
   - Assessment Methods
   - Course Topics

2. Development Environment Setup
   - Required Hardware
   - Required Software
   - Installation Steps

---

## Outline (2/3)

3. Introduction to Parallel Programming
   - What is Parallel Programming?
   - Why Parallel Programming?
   - Basic Concepts

4. First Parallel Program
   - Hello World Example
   - Compilation Steps
   - Running and Testing

---

## Outline (3/3)

5. Understanding Hardware
   - CPU Architecture
   - Memory Patterns

6. Performance and Practice
   - Parallel Patterns
   - Performance Measurement
   - Homework
   - Resources

---

## 1. Course Overview

### Course Description

This course introduces fundamental concepts and practices of parallel programming, focusing on:
- Designing and implementing efficient parallel algorithms
- Using modern programming frameworks
- Understanding parallel architectures
- Analyzing and optimizing parallel programs

---

### Learning Outcomes (1/2)

After completing this course, you will be able to:

1. Design and implement parallel algorithms using OpenMP and MPI
2. Analyze and optimize parallel program performance
3. Develop solutions using various programming models

---

### Learning Outcomes (2/2)

4. Apply parallel computing concepts to real-world problems
5. Evaluate and select appropriate parallel computing approaches based on:
   - Problem requirements
   - Hardware constraints
   - Performance goals

---

### Assessment Methods

| Assessment                | Weight | Due Date |
|--------------------------|---------|----------|
| Midterm Project Report   | 60%     | Week 8   |
| Quiz-1                   | 40%     | Week 7   |
| Final Project Report     | 70%     | Week 14  |
| Quiz-2                   | 30%     | Week 13  |

---

### Course Topics (1/2)

1. Parallel computing concepts
   - Basic principles
   - Architecture overview
   - Programming models

2. Algorithm design and analysis
   - Design patterns
   - Performance metrics
   - Optimization strategies

---

### Course Topics (2/2)

3. Programming frameworks
   - OpenMP
   - MPI
   - GPU Computing

4. Advanced topics
   - Performance optimization
   - Real-world applications
   - Best practices

---

### Why Parallel Programming? (1/2)

#### Historical Evolution
- Moore's Law limitations
- Multi-core revolution
- Cloud computing era
- Big data requirements

#### Industry Applications
- Scientific simulations
- Financial modeling
- AI/Machine Learning
- Video processing

---

### Why Parallel Programming? (2/2)

#### Performance Benefits
- Reduced execution time
- Better resource utilization
- Improved responsiveness
- Higher throughput

#### Challenges
- Synchronization overhead
- Load balancing
- Debugging complexity
- Race conditions

---

### Parallel Computing Models (1/2)

#### Shared Memory
```text
CPU     CPU     CPU     CPU
  │       │       │       │
  └───────┴───────┴───────┘
          │
    Shared Memory
```

- All processors access same memory
- Easy to program
- Limited scalability
- Example: OpenMP

---

### Parallel Computing Models (2/2)

#### Distributed Memory
```text
CPU──Memory   CPU──Memory
    │             │
    └─────Network─┘
    │             │
CPU──Memory   CPU──Memory
```

- Each processor has private memory
- Better scalability
- More complex programming
- Example: MPI

---

### Memory Architecture Deep Dive (1/3)

#### Cache Hierarchy
```cpp
// Example showing cache effects
void demonstrateCacheEffects() {
    const int SIZE = 1024 * 1024;
    int* arr = new int[SIZE];
    
    // Sequential access (cache-friendly)
    Timer t1;
    for(int i = 0; i < SIZE; i++) {
        arr[i] = i;
    }
    double sequential_time = t1.elapsed();
    
    // Random access (cache-unfriendly)
    Timer t2;
    for(int i = 0; i < SIZE; i++) {
        arr[(i * 16) % SIZE] = i;
    }
    double random_time = t2.elapsed();
    
    printf("Sequential/Random time ratio: %f\n", 
           random_time/sequential_time);
}
```

---

### Memory Architecture Deep Dive (2/3)

#### False Sharing Example
```cpp
#include <omp.h>

// Bad example with false sharing
void falseSharing() {
    int data[4];
    #pragma omp parallel for
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 1000000; j++) {
            data[i]++; // Adjacent elements share cache line
        }
    }
}

// Better version avoiding false sharing
void avoidFalseSharing() {
    struct PaddedInt {
        int value;
        char padding[60]; // Separate cache lines
    };
    PaddedInt data[4];
    
    #pragma omp parallel for
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 1000000; j++) {
            data[i].value++;
        }
    }
}
```

---

### Memory Architecture Deep Dive (3/3)

#### NUMA Awareness
```cpp
// NUMA-aware allocation
void numaAwareAllocation() {
    #pragma omp parallel
    {
        // Each thread allocates its own memory
        std::vector<double> local_data(1000000);
        
        // Process local data
        #pragma omp for
        for(int i = 0; i < local_data.size(); i++) {
            local_data[i] = heavyComputation(i);
        }
    }
}
```

---

### OpenMP Fundamentals (1/4)

#### Basic Parallel Regions
```cpp
#include <omp.h>

void basicParallelRegion() {
    #pragma omp parallel
    {
        // This code runs in parallel
        int thread_id = omp_get_thread_num();
        
        #pragma omp critical
        std::cout << "Thread " << thread_id << " starting\n";
        
        // Do some work
        heavyComputation();
        
        #pragma omp critical
        std::cout << "Thread " << thread_id << " finished\n";
    }
}
```

---

### OpenMP Fundamentals (2/4)

#### Work Sharing Constructs
```cpp
void workSharing() {
    const int SIZE = 1000000;
    std::vector<double> data(SIZE);
    
    // Parallel for loop
    #pragma omp parallel for schedule(dynamic, 1000)
    for(int i = 0; i < SIZE; i++) {
        data[i] = heavyComputation(i);
    }
    
    // Parallel sections
    #pragma omp parallel sections
    {
        #pragma omp section
        { task1(); }
        
        #pragma omp section
        { task2(); }
    }
}
```

---

### OpenMP Fundamentals (3/4)

#### Data Sharing
```cpp
void dataSharing() {
    int shared_var = 0;
    int private_var = 0;
    
    #pragma omp parallel private(private_var) \
                         shared(shared_var)
    {
        private_var = omp_get_thread_num(); // Each thread has its copy
        
        #pragma omp critical
        shared_var += private_var; // Updates shared variable
    }
}
```

---

### OpenMP Fundamentals (4/4)

#### Synchronization
```cpp
void synchronization() {
    #pragma omp parallel
    {
        // Barrier synchronization
        #pragma omp barrier
        
        // Critical section
        #pragma omp critical
        {
            // Exclusive access
        }
        
        // Atomic operation
        #pragma omp atomic
        counter++;
    }
}
```

---

### Practical Workshop (1/3)

#### Matrix Multiplication
```cpp
void matrixMultiply(const std::vector<std::vector<double>>& A,
                   const std::vector<std::vector<double>>& B,
                   std::vector<std::vector<double>>& C) {
    int N = A.size();
    
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            double sum = 0.0;
            for(int k = 0; k < N; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}
```

---

### Practical Workshop (2/3)

#### Performance Comparison
```cpp
void comparePerformance() {
    const int N = 1000;
    auto A = generateRandomMatrix(N);
    auto B = generateRandomMatrix(N);
    auto C1 = createEmptyMatrix(N);
    auto C2 = createEmptyMatrix(N);
    
    // Sequential version
    Timer t1;
    matrixMultiplySequential(A, B, C1);
    double sequential_time = t1.elapsed();
    
    // Parallel version
    Timer t2;
    matrixMultiply(A, B, C2);
    double parallel_time = t2.elapsed();
    
    printf("Speedup: %f\n", sequential_time/parallel_time);
}
```

---

### Practical Workshop (3/3)

#### Exercise Tasks
1. Implement matrix multiplication
2. Measure performance with different matrix sizes
3. Try different scheduling strategies
4. Plot performance results

---

## 2. Development Environment

### Required Hardware

- Multi-core processor
- 16GB RAM (recommended)
- 100GB free disk space
- Windows 10/11 (version 2004+)

---

### Required Software

1. Visual Studio Community 2022
2. Windows Subsystem for Linux (WSL2)
3. Ubuntu distribution
4. Git for Windows

---

### Step-by-Step Installation Guide

#### Windows Installation (30 minutes)

1. **Visual Studio Code Installation**
   - Go to [Visual Studio Code](https://code.visualstudio.com/)
   - Click "Download for Windows"
   - Run the installer (VSCodeUserSetup-x64-*.exe)
   - ✅ Check "Add to PATH" during installation
   - ✅ Check "Add 'Open with Code' action"

2. **MinGW Compiler Installation**
   ```bash
   # Step 1: Download MSYS2
   # Visit https://www.msys2.org/ and download installer
   
   # Step 2: Run MSYS2 installer
   # Use default installation path: C:\msys64
   
   # Step 3: Open MSYS2 terminal and run:
   pacman -Syu  # Update package database
   # Close terminal when asked
   
   # Step 4: Reopen MSYS2 and install required packages:
   pacman -S mingw-w64-x86_64-gcc
   pacman -S mingw-w64-x86_64-gdb
   ```

3. **Add to PATH**
   - Open Windows Search
   - Type "Environment Variables"
   - Click "Edit the system environment variables"
   - Click "Environment Variables"
   - Under "System Variables", find "Path"
   - Click "Edit" → "New"
   - Add `C:\msys64\mingw64\bin`
   - Click "OK" on all windows

4. **Verify Installation**
   ```bash
   # Open new Command Prompt and type:
   gcc --version
   g++ --version
   gdb --version
   ```

#### VS Code Configuration (15 minutes)

1. **Install Required Extensions**
   - Open VS Code
   - Press Ctrl+Shift+X
   - Install these extensions:
     - C/C++ Extension Pack
     - Code Runner
     - GitLens
     - Live Share

2. **Create Workspace**
   ```bash
   # Open Command Prompt
   mkdir parallel_programming
   cd parallel_programming
   code .
   ```

3. **Configure Build Tasks**
   - Press Ctrl+Shift+P
   - Type "Tasks: Configure Default Build Task"
   - Select "Create tasks.json from template"
   - Select "Others"
   - Replace content with:
   ```json
   {
       "version": "2.0.0",
       "tasks": [
           {
               "label": "build",
               "type": "shell",
               "command": "g++",
               "args": [
                   "-g",
                   "-fopenmp",
                   "${file}",
                   "-o",
                   "${fileDirname}/${fileBasenameNoExtension}"
               ],
               "group": {
                   "kind": "build",
                   "isDefault": true
               }
           }
       ]
   }
   ```

#### First OpenMP Program (15 minutes)

1. **Create Test File**
   - In VS Code, create new file: `test.cpp`
   - Add this code:
   ```cpp
   #include <iostream>
   #include <omp.h>
   
   int main() {
       // Get total available threads
       int max_threads = omp_get_max_threads();
       printf("System has %d processors available\n", max_threads);
       
       // Set number of threads
       omp_set_num_threads(4);
       
       // Parallel region
       #pragma omp parallel
       {
           int id = omp_get_thread_num();
           printf("Hello from thread %d\n", id);
           
           // Only master thread prints total
           if (id == 0) {
               printf("Total %d threads running\n", 
                      omp_get_num_threads());
           }
       }
       return 0;
   }
   ```

2. **Compile and Run**
   - Press Ctrl+Shift+B to build
   - Open terminal (Ctrl+`)
   - Run program:
   ```bash
   ./test
   ```

3. **Experiment**
   ```bash
   # Try different thread counts
   set OMP_NUM_THREADS=2
   ./test
   
   set OMP_NUM_THREADS=8
   ./test
   ```

#### Common Issues and Solutions

1. **Compiler Not Found**
   - Verify PATH setting
   - Restart VS Code
   - Restart Command Prompt

2. **OpenMP Not Recognized**
   - Ensure `-fopenmp` flag in tasks.json
   - Check compiler version supports OpenMP

3. **Program Crashes**
   - Check array bounds
   - Verify thread synchronization
   - Use proper reduction clauses

#### Practice Exercises

1. **Basic Parallel For**
   ```cpp
   // Create array_sum.cpp
   #include <omp.h>
   #include <vector>
   
   int main() {
       const int SIZE = 1000000;
       std::vector<int> data(SIZE);
       long sum = 0;
       
       // Initialize array
       for(int i = 0; i < SIZE; i++) {
           data[i] = i;
       }
       
       // Parallel sum
       #pragma omp parallel for reduction(+:sum)
       for(int i = 0; i < SIZE; i++) {
           sum += data[i];
       }
       
       printf("Sum: %ld\n", sum);
       return 0;
   }
   ```

2. **Thread Private Data**
   ```cpp
   // Create thread_private.cpp
   #include <omp.h>
   
   int main() {
       int thread_sum = 0;
       
       #pragma omp parallel private(thread_sum)
       {
           thread_sum = omp_get_thread_num();
           printf("Thread %d: sum = %d\n", 
                  omp_get_thread_num(), thread_sum);
       }
       
       printf("Final sum: %d\n", thread_sum);
       return 0;
   }
   ```

---

## 3. Introduction to Parallel Programming

### What is Parallel Programming? (1/2)

Parallel programming is the technique of writing programs that:
- Execute multiple tasks simultaneously
- Utilize multiple computational resources
- Improve performance through parallelization

---

### What is Parallel Programming? (2/2)

Key Concepts:
- Task decomposition
- Data distribution
- Load balancing
- Synchronization

---

## 4. First Parallel Program

### Hello World Example

```cpp
#include <iostream>
#include <omp.h>

int main() {
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int total_threads = omp_get_num_threads();
        
        printf("Hello from thread %d of %d!\n", 
               thread_id, total_threads);
    }
    return 0;
}
```

---

### Compilation Steps

**Visual Studio:**
```bash
# Create new project
mkdir parallel_hello
cd parallel_hello

# Compile with OpenMP
cl /openmp hello.cpp
```

---

### Running and Testing

**Windows:**
```bash
hello.exe
```

**Linux/WSL:**
```bash
./hello
```

Expected Output:
```
Hello from thread 0 of 4!
Hello from thread 2 of 4!
Hello from thread 3 of 4!
Hello from thread 1 of 4!
```

---

## 5. Understanding Hardware

### CPU Architecture

```text
CPU
├── Core 0
│   ├── L1 Cache
│   └── L2 Cache
├── Core 1
│   ├── L1 Cache
│   └── L2 Cache
└── Shared L3 Cache
```

---

### Memory Access Patterns

```cpp
void measureMemoryAccess() {
    const int SIZE = 1000000;
    std::vector<int> data(SIZE);
    
    // Sequential access
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < SIZE; i++) {
        data[i] = i;
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    // Random access
    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < SIZE; i++) {
        data[(i * 16) % SIZE] = i;
    }
    end = std::chrono::high_resolution_clock::now();
}
```

---

## 6. Parallel Patterns

### Data Parallelism Example

```cpp
#include <omp.h>
#include <vector>

void vectorAdd(const std::vector<int>& a, 
               const std::vector<int>& b, 
               std::vector<int>& result) {
    #pragma omp parallel for
    for(int i = 0; i < a.size(); i++) {
        result[i] = a[i] + b[i];
    }
}
```

---

### Task Parallelism Example

```cpp
#pragma omp parallel sections
{
    #pragma omp section
    {
        // Task 1: Matrix multiplication
    }
    
    #pragma omp section
    {
        // Task 2: File processing
    }
}
```

---

## 7. Performance Measurement

### Using the Timer Class

```cpp
class Timer {
    std::chrono::high_resolution_clock::time_point start;
public:
    Timer() : start(std::chrono::high_resolution_clock::now()) {}
    
    double elapsed() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end - start).count();
    }
};
```

---

### Measuring Parallel Performance

```cpp
void measureParallelPerformance() {
    const int SIZE = 100000000;
    std::vector<double> data(SIZE);
    
    Timer t;
    #pragma omp parallel for
    for(int i = 0; i < SIZE; i++) {
        data[i] = std::sin(i) * std::cos(i);
    }
    std::cout << "Time: " << t.elapsed() << "s\n";
}
```

---

## 8. Homework

### Assignment 1: Environment Setup
1. Screenshots of installations
2. Version information
3. Example program results
4. Issue resolution documentation

---

### Assignment 2: Performance Analysis
1. Process & thread ID printing
2. Execution time measurements
3. Performance graphs
4. Analysis report

---

## 9. Resources

### Documentation
- OpenMP API Specification
- Visual Studio Parallel Programming
- WSL Documentation

### Books and Tutorials
- "Introduction to Parallel Programming"
- "Using OpenMP"
- Online courses

---

## Next Week Preview

We will cover:
- Advanced parallel patterns
- Performance analysis
- OpenMP features
- Practical exercises

---

## 10. Advanced OpenMP Features

### Nested Parallelism (1/2)

```cpp
#include <omp.h>

void nestedParallelExample() {
    omp_set_nested(1); // Enable nested parallelism
    
    #pragma omp parallel num_threads(2)
    {
        int outer_id = omp_get_thread_num();
        
        #pragma omp parallel num_threads(2)
        {
            int inner_id = omp_get_thread_num();
            printf("Outer thread %d, Inner thread %d\n", 
                   outer_id, inner_id);
        }
    }
}
```

---

### Nested Parallelism (2/2)

Expected Output:
```
Outer thread 0, Inner thread 0
Outer thread 0, Inner thread 1
Outer thread 1, Inner thread 0
Outer thread 1, Inner thread 1
```

Benefits:
- Hierarchical parallelism
- Better resource utilization
- Complex parallel patterns

---

### Task-Based Parallelism (1/3)

```cpp
void taskBasedExample() {
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            heavyTask1();
            
            #pragma omp task
            heavyTask2();
            
            #pragma omp taskwait
            printf("All tasks completed\n");
        }
    }
}
```

---

### Task-Based Parallelism (2/3)

#### Fibonacci Example
```cpp
int parallel_fib(int n) {
    if (n < 30) return fib_sequential(n);
    
    int x, y;
    #pragma omp task shared(x)
    x = parallel_fib(n - 1);
    
    #pragma omp task shared(y)
    y = parallel_fib(n - 2);
    
    #pragma omp taskwait
    return x + y;
}
```

---

### Task-Based Parallelism (3/3)

#### Task Priority
```cpp
void priorityTasks() {
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task priority(0)
            lowPriorityTask();
            
            #pragma omp task priority(100)
            highPriorityTask();
        }
    }
}
```

---

## 11. Performance Optimization Techniques

### Loop Optimization (1/3)

#### Loop Scheduling
```cpp
void demonstrateScheduling() {
    const int SIZE = 1000000;
    
    // Static scheduling
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < SIZE; i++)
        work_static(i);
        
    // Dynamic scheduling
    #pragma omp parallel for schedule(dynamic, 1000)
    for(int i = 0; i < SIZE; i++)
        work_dynamic(i);
        
    // Guided scheduling
    #pragma omp parallel for schedule(guided)
    for(int i = 0; i < SIZE; i++)
        work_guided(i);
}
```

---

### Loop Optimization (2/3)

#### Loop Collapse
```cpp
void matrixOperations() {
    const int N = 1000;
    double matrix[N][N];
    
    // Without collapse
    #pragma omp parallel for
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++)
            matrix[i][j] = compute(i, j);
            
    // With collapse
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++)
            matrix[i][j] = compute(i, j);
}
```

---

### Loop Optimization (3/3)

#### SIMD Directives
```cpp
void simdExample() {
    const int N = 1000000;
    float a[N], b[N], c[N];
    
    #pragma omp parallel for simd
    for(int i = 0; i < N; i++) {
        c[i] = a[i] * b[i];
    }
}
```

---

## 12. Common Parallel Programming Patterns

### Pipeline Pattern (1/2)

```cpp
struct Data {
    // ... data members
};

void pipelinePattern() {
    std::queue<Data> queue1, queue2;
    
    #pragma omp parallel sections
    {
        #pragma omp section // Stage 1
        {
            while(hasInput()) {
                Data d = readInput();
                queue1.push(d);
            }
        }
        
        #pragma omp section // Stage 2
        {
            while(true) {
                Data d = queue1.pop();
                process(d);
                queue2.push(d);
            }
        }
        
        #pragma omp section // Stage 3
        {
            while(true) {
                Data d = queue2.pop();
                writeOutput(d);
            }
        }
    }
}
```

---

### Pipeline Pattern (2/2)

Benefits:
- Improved throughput
- Better resource utilization
- Natural for streaming data

Challenges:
- Load balancing
- Queue management
- Termination conditions

---

## 13. Debugging Parallel Programs

### Common Issues (1/2)

1. Race Conditions
```cpp
// Bad code
int counter = 0;
#pragma omp parallel for
for(int i = 0; i < N; i++)
    counter++; // Race condition!

// Fixed code
int counter = 0;
#pragma omp parallel for reduction(+:counter)
for(int i = 0; i < N; i++)
    counter++;
```

---

### Common Issues (2/2)

2. Deadlocks
```cpp
// Potential deadlock
#pragma omp parallel sections
{
    #pragma omp section
    {
        #pragma omp critical(A)
        {
            #pragma omp critical(B)
            { /* ... */ }
        }
    }
    
    #pragma omp section
    {
        #pragma omp critical(B)
        {
            #pragma omp critical(A)
            { /* ... */ }
        }
    }
}
```

---

## 14. Real-World Applications

### Image Processing Example

```cpp
void parallelImageProcessing(unsigned char* image, 
                           int width, int height) {
    #pragma omp parallel for collapse(2)
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            
            // Apply Gaussian blur
            float sum_r = 0, sum_g = 0, sum_b = 0;
            float weight_sum = 0;
            
            for(int ky = -2; ky <= 2; ky++) {
                for(int kx = -2; kx <= 2; kx++) {
                    int ny = y + ky;
                    int nx = x + kx;
                    
                    if(ny >= 0 && ny < height && 
                       nx >= 0 && nx < width) {
                        float weight = gaussian(kx, ky);
                        int nidx = (ny * width + nx) * 3;
                        
                        sum_r += image[nidx + 0] * weight;
                        sum_g += image[nidx + 1] * weight;
                        sum_b += image[nidx + 2] * weight;
                        weight_sum += weight;
                    }
                }
            }
            
            // Store result
            image[idx + 0] = sum_r / weight_sum;
            image[idx + 1] = sum_g / weight_sum;
            image[idx + 2] = sum_b / weight_sum;
        }
    }
}
```

---

### Monte Carlo Simulation

```cpp
double parallelMonteCarlo(int iterations) {
    long inside_circle = 0;
    
    #pragma omp parallel reduction(+:inside_circle)
    {
        unsigned int seed = omp_get_thread_num();
        
        #pragma omp for
        for(int i = 0; i < iterations; i++) {
            double x = (double)rand_r(&seed) / RAND_MAX;
            double y = (double)rand_r(&seed) / RAND_MAX;
            
            if(x*x + y*y <= 1.0)
                inside_circle++;
        }
    }
    
    return 4.0 * inside_circle / iterations;
}
```

---

## 15. Advanced Workshop

### Project: Parallel Sort Implementation

1. Sequential Quicksort
2. Parallel Quicksort
3. Performance Comparison
4. Visualization Tools

---

### Workshop Tasks (1/3)

```cpp
// Sequential Quicksort
void quicksort(int* arr, int left, int right) {
    if(left < right) {
        int pivot = partition(arr, left, right);
        quicksort(arr, left, pivot - 1);
        quicksort(arr, pivot + 1, right);
    }
}
```

---

### Workshop Tasks (2/3)

```cpp
// Parallel Quicksort
void parallel_quicksort(int* arr, int left, int right) {
    if(left < right) {
        if(right - left < THRESHOLD) {
            quicksort(arr, left, right);
            return;
        }
        
        int pivot = partition(arr, left, right);
        
        #pragma omp task
        parallel_quicksort(arr, left, pivot - 1);
        
        #pragma omp task
        parallel_quicksort(arr, pivot + 1, right);
        
        #pragma omp taskwait
    }
}
```

---

### Workshop Tasks (3/3)

Performance Analysis Tools:
```cpp
void analyzePerformance() {
    const int SIZES[] = {1000, 10000, 100000, 1000000};
    const int THREADS[] = {1, 2, 4, 8, 16};
    
    for(int size : SIZES) {
        for(int threads : THREADS) {
            omp_set_num_threads(threads);
            
            // Run and measure
            auto arr = generateRandomArray(size);
            Timer t;
            
            #pragma omp parallel
            {
                #pragma omp single
                parallel_quicksort(arr.data(), 0, size-1);
            }
            
            double time = t.elapsed();
            printf("Size: %d, Threads: %d, Time: %f\n",
                   size, threads, time);
        }
    }
}
```

---

## Cross-Platform Development Environment (1/5)

### Project Template

Download or clone the template project:
```bash
git clone https://github.com/ucoruh/cpp-openmp-template
# or create manually:
mkdir parallel-programming
cd parallel-programming
```

Create this structure:
```
parallel-programming/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   └── include/
│       └── config.h
├── build/
│   ├── windows/
│   └── linux/
└── scripts/
    ├── build-windows.bat
    └── build-linux.sh
```

---

## Cross-Platform Development Environment (2/5)

### CMakeLists.txt
```cmake
cmake_minimum_required(VERSION 3.15)
project(parallel-programming)

# C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
else()
    message(FATAL_ERROR "OpenMP not found")
endif()

# Add executable
add_executable(${PROJECT_NAME} 
    src/main.cpp
)

# Include directories
target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)

# Link OpenMP
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        OpenMP::OpenMP_CXX
)
```

---

## Cross-Platform Development Environment (3/5)

### Build Scripts

**build-windows.bat:**
```batch
@echo off
setlocal

:: Create build directory
mkdir build\windows 2>nul
cd build\windows

:: CMake configuration
cmake -G "Visual Studio 17 2022" -A x64 ..\..

:: Debug build
cmake --build . --config Debug

:: Release build
cmake --build . --config Release

cd ..\..

echo Build completed!
pause
```

**build-linux.sh:**
```bash
#!/bin/bash

# Create build directory
mkdir -p build/linux
cd build/linux

# CMake configuration
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ../..

# Build
ninja

cd ../..

echo "Build completed!"
```

---

## Cross-Platform Development Environment (4/5)

### Platform-Independent Code

**config.h:**
```cpp
#pragma once

// Platform check
#if defined(_WIN32)
    #define PLATFORM_WINDOWS
#elif defined(__linux__)
    #define PLATFORM_LINUX
#else
    #error "Unsupported platform"
#endif

// OpenMP check
#ifdef _OPENMP
    #define HAVE_OPENMP
#endif
```

**main.cpp:**
```cpp
#include <iostream>
#include <vector>
#include <omp.h>
#include "config.h"

int main() {
    // OpenMP version check
    #ifdef _OPENMP
        std::cout << "OpenMP Version: " 
                  << _OPENMP << std::endl;
    #else
        std::cout << "OpenMP not supported" << std::endl;
        return 1;
    #endif

    // Set thread count
    omp_set_num_threads(4);

    // Parallel region
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int total_threads = omp_get_num_threads();
        
        #pragma omp critical
        {
            std::cout << "Thread " << thread_id 
                      << " of " << total_threads 
                      << std::endl;
        }
    }

    return 0;
}
```

---

## Cross-Platform Development Environment (5/5)

### Common Issues and Solutions

1. **CMake OpenMP Issues:**
   - Windows: Reinstall Visual Studio
   - Linux: `sudo apt install libomp-dev`

2. **WSL Connection Issues:**
   ```powershell
   wsl --shutdown
   wsl --update
   ```

3. **Build Errors:**
   - Delete build directory
   - Delete CMakeCache.txt
   - Rebuild project

4. **VS2022 WSL Target Missing:**
   - Run VS2022 as administrator
   - Install Linux Development workload
   - Restart WSL

---

## Additional Resources

- [Visual Studio Documentation](https://docs.microsoft.com/en-us/visualstudio/)
- [WSL Documentation](https://docs.microsoft.com/en-us/windows/wsl/)
- [CMake Tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/)
- [OpenMP Documentation](https://www.openmp.org/resources/)

For questions and help:
- GitHub Issues
- Email
- Office hours

$$
End-Of-Week-1
$$
