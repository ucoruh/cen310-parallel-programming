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
header: 'CEN310 Paralel Programlama Hafta-3'
footer: '![height:50px](http://erdogan.edu.tr/Images/Uploads/MyContents/L_379-20170718142719217230.jpg) RTEÜ CEN310 Hafta-3'
title: "CEN310 Paralel Programlama Hafta-3"
author: "Öğr. Gör. Dr. Uğur CORUH"
date:
subtitle: "OpenMP Programlama"
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

## Hafta-3

#### OpenMP Programlama

---

## Ders İçeriği

1. OpenMP'ye Giriş
   - OpenMP Nedir?
   - Fork-Join Modeli
   - Derleyici Direktifleri
   - Çalışma Zamanı Kütüphane Fonksiyonları
   - Ortam Değişkenleri

2. OpenMP Direktifleri
   - Paralel Bölgeler
   - İş Paylaşımı Yapıları
   - Veri Paylaşım Özellikleri
   - Senkronizasyon

3. OpenMP Programlama Örnekleri
   - Temel Paralel Döngüler
   - İndirgeme İşlemleri
   - Görev Paralelliği
   - İç İçe Paralellik

4. Performans Değerlendirmeleri
   - İş Parçacığı Yönetimi
   - Yük Dengeleme
   - Veri Yerleşimi
   - Önbellek Etkileri

---

## 1. OpenMP'ye Giriş

### OpenMP Nedir?

- Paylaşımlı bellek paralel programlama için API
- C, C++ ve Fortran desteği
- Derleyici direktiflerine dayalı
- Taşınabilir ve ölçeklenebilir

Örnek:
```cpp
#include <omp.h>

int main() {
    #pragma omp parallel
    {
        printf("%d numaralı iş parçacığından merhaba\n", 
               omp_get_thread_num());
    }
    return 0;
}
```

---

// ... Week-3 için detaylı içerik devam edecek 