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
header: 'CEN310 Paralel Programlama Hafta-5'
footer: '![height:50px](http://erdogan.edu.tr/Images/Uploads/MyContents/L_379-20170718142719217230.jpg) RTEÜ CEN310 Hafta-5'
title: "CEN310 Paralel Programlama Hafta-5"
author: "Öğr. Gör. Dr. Uğur CORUH"
date:
subtitle: "GPU Programlama"
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

## Hafta-5

#### GPU Programlama

---

## Ders İçeriği

1. GPU Hesaplamaya Giriş
   - GPU Mimarisi Genel Bakış
   - CUDA Programlama Modeli
   - GPU Bellek Hiyerarşisi
   - İş Parçacığı Hiyerarşisi
   - Çekirdek Fonksiyonlar

2. CUDA Programlama Temelleri
   - Bellek Yönetimi
   - İş Parçacığı Organizasyonu
   - Senkronizasyon
   - Hata Yönetimi
   - CUDA Çalışma Zamanı API'si

3. Performans Optimizasyonu
   - Bellek Birleştirme
   - Paylaşımlı Bellek Kullanımı
   - Bank Çakışmaları
   - Doluluk Oranı
   - Warp Sapması

4. İleri GPU Programlama
   - Akışlar ve Olaylar
   - Asenkron İşlemler
   - Çoklu-GPU Programlama
   - GPU-CPU Veri Transferi
   - Birleşik Bellek

---

## 1. GPU Hesaplamaya Giriş

### GPU Mimarisi

```text
Ana Bilgisayar (CPU)    Cihaz (GPU)
        ↓                    ↓
     Bellek            Global Bellek
        ↓                    ↓
      ←--- PCI Express Yolu -→
```

Temel Kavramlar:
- Yoğun paralel mimari
- Binlerce çekirdek
- SIMT yürütme modeli
- Bellek hiyerarşisi

Örnek CUDA Programı:
```cpp
#include <cuda_runtime.h>

__global__ void vektorTopla(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 1000000;
    size_t boyut = n * sizeof(float);
    
    // Ana bilgisayar belleği ayırma
    float *h_a = (float*)malloc(boyut);
    float *h_b = (float*)malloc(boyut);
    float *h_c = (float*)malloc(boyut);
    
    // Dizileri başlatma
    for(int i = 0; i < n; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }
    
    // GPU belleği ayırma
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, boyut);
    cudaMalloc(&d_b, boyut);
    cudaMalloc(&d_c, boyut);
    
    // GPU'ya kopyalama
    cudaMemcpy(d_a, h_a, boyut, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, boyut, cudaMemcpyHostToDevice);
    
    // Çekirdeği başlatma
    int blokBoyutu = 256;
    int blokSayisi = (n + blokBoyutu - 1) / blokBoyutu;
    vektorTopla<<<blokSayisi, blokBoyutu>>>(d_a, d_b, d_c, n);
    
    // Sonucu geri kopyalama
    cudaMemcpy(h_c, d_c, boyut, cudaMemcpyDeviceToHost);
    
    // Temizlik
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}
```

---

// ... Hafta-5 için detaylı içerik devam edecek 