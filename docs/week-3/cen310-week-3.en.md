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
header: 'CEN310 Parallel Programming Week-3'
footer: '![height:50px](http://erdogan.edu.tr/Images/Uploads/MyContents/L_379-20170718142719217230.jpg) RTEU CEN310 Week-3'
title: "CEN310 Parallel Programming Week-3"
author: "Author: Dr. Uğur CORUH"
date:
subtitle: "OpenMP Programming"
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

## Week-3

#### OpenMP Programming

---

## Outline

1. Introduction to OpenMP
   - What is OpenMP?
   - Fork-Join Model
   - Compiler Directives
   - Runtime Library Functions
   - Environment Variables

2. OpenMP Directives
   - Parallel Regions
   - Work Sharing Constructs
   - Data Sharing Attributes
   - Synchronization

3. OpenMP Programming Examples
   - Basic Parallel Loops
   - Reduction Operations
   - Task Parallelism
   - Nested Parallelism

4. Performance Considerations
   - Thread Management
   - Load Balancing
   - Data Locality
   - Cache Effects

---

## 1. Introduction to OpenMP

### What is OpenMP?

- API for shared-memory parallel programming
- Supports C, C++, and Fortran
- Based on compiler directives
- Portable and scalable

Example:
```cpp
#include <omp.h>

int main() {
    #pragma omp parallel
    {
        printf("Hello from thread %d\n", 
               omp_get_thread_num());
    }
    return 0;
}
```

---

// ... continue with detailed content for Week-3 