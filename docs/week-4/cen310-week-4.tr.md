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
header: 'CEN310 Paralel Programlama Hafta-4'
footer: '![height:50px](http://erdogan.edu.tr/Images/Uploads/MyContents/L_379-20170718142719217230.jpg) RTEÜ CEN310 Hafta-4'
title: "CEN310 Paralel Programlama Hafta-4"
author: "Öğr. Gör. Dr. Uğur CORUH"
date:
subtitle: "MPI Programlama"
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
footer-center: "Lisans: CC BY-NC-ND 4.0"
footer-right:
subparagraph: true
lang: tr-TR
math: katex
---

<!-- _backgroundColor: aquq -->

<!-- _color: orange -->

<!-- paginate: false -->

# CEN310 Paralel Programlama

## Hafta-4

#### MPI Programlama

---

## Ders İçeriği

1. MPI'ya Giriş
   - MPI Nedir?
   - Dağıtık Bellek Modeli
   - MPI Uygulama Türleri
   - Temel Kavramlar
   - Ortam Kurulumu

2. Noktadan Noktaya İletişim
   - Engelleyici Gönderme/Alma
   - Engelleyici Olmayan Gönderme/Alma
   - Tamponlama ve Senkronizasyon
   - İletişim Modları
   - Hata Yönetimi

3. Toplu İletişim
   - Yayın (Broadcast)
   - Dağıtma/Toplama (Scatter/Gather)
   - İndirgeme İşlemleri
   - Tümünden Tümüne İletişim
   - Bariyerler

4. İleri Seviye MPI Özellikleri
   - Türetilmiş Veri Tipleri
   - Sanal Topolojiler
   - Tek Taraflı İletişim
   - Hibrit Programlama (MPI + OpenMP)

---

## 1. MPI'ya Giriş

### MPI Nedir?

- Mesaj Geçirme Arayüzü standardı
- Platform-bağımsız iletişim protokolü
- Dağıtık bellek sistemlerini destekler
- C, C++, Fortran dil bağlantıları

Örnek:
```cpp
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    printf("Süreç %d / %d\n", rank, size);
    
    MPI_Finalize();
    return 0;
}
```

---

// ... Hafta-4 için detaylı içerik devam edecek 