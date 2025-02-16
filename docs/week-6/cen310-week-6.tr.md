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
header: 'CEN310 Paralel Programlama Hafta-6'
footer: '![height:50px](http://erdogan.edu.tr/Images/Uploads/MyContents/L_379-20170718142719217230.jpg) RTEÜ CEN310 Hafta-6'
title: "CEN310 Paralel Programlama Hafta-6"
author: "Öğr. Gör. Dr. Uğur CORUH"
date:
subtitle: "Performans Optimizasyonu"
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

## Hafta-6

#### Performans Optimizasyonu

---

## Ders İçeriği

1. Performans Analiz Araçları
   - Profil Çıkarıcılar
   - Donanım Sayaçları
   - Performans Metrikleri
   - Darboğaz Tespiti
   - Kıyaslama

2. Bellek Optimizasyonu
   - Önbellek Optimizasyonu
   - Bellek Erişim Desenleri
   - Veri Yerleşimi
   - Yanlış Paylaşım
   - Bellek Bant Genişliği

3. Algoritma Optimizasyonu
   - Yük Dengeleme
   - İş Dağıtımı
   - İletişim Desenleri
   - Senkronizasyon Yükü
   - Ölçeklenebilirlik Analizi

4. İleri Optimizasyon Teknikleri
   - Vektörleştirme
   - Döngü Optimizasyonu
   - İş Parçacığı İlişkilendirme
   - Derleyici Optimizasyonları
   - Donanıma Özel Ayarlamalar

---

## 1. Performans Analiz Araçları

### Profil Çıkarıcı Kullanımı

Intel VTune örneği:
```cpp
#include <omp.h>
#include <vector>

void matris_carpimi_optimize(const std::vector<float>& A,
                           const std::vector<float>& B,
                           std::vector<float>& C,
                           int N) {
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            float toplam = 0.0f;
            // Önbellek dostu erişim deseni
            for(int k = 0; k < N; k++) {
                toplam += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = toplam;
        }
    }
}

// Performans ölçümü
void performans_olc() {
    const int N = 1024;
    std::vector<float> A(N * N), B(N * N), C(N * N);
    
    // Matrisleri başlat
    for(int i = 0; i < N * N; i++) {
        A[i] = rand() / (float)RAND_MAX;
        B[i] = rand() / (float)RAND_MAX;
    }
    
    double baslangic = omp_get_wtime();
    matris_carpimi_optimize(A, B, C, N);
    double bitis = omp_get_wtime();
    
    printf("Süre: %f saniye\n", bitis - baslangic);
}
```

---

## 2. Bellek Optimizasyonu

### Önbellek Dostu Veri Erişimi

```cpp
// Kötü: Önbellek dostu olmayan erişim
void kotu_erisim(float* matris, int N) {
    for(int j = 0; j < N; j++) {
        for(int i = 0; i < N; i++) {
            matris[i * N + j] = hesapla(i, j);
        }
    }
}

// İyi: Önbellek dostu erişim
void iyi_erisim(float* matris, int N) {
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            matris[i * N + j] = hesapla(i, j);
        }
    }
}
```

---

### Yanlış Paylaşım Önleme

```cpp
// Kötü: Yanlış paylaşım
struct KotuSayac {
    int sayac;  // Birden çok iş parçacığı bitişik belleği güncelliyor
};

// İyi: Yanlış paylaşımı önlemek için dolgu
struct IyiSayac {
    int sayac;
    char dolgu[60];  // Önbellek satırı boyutuna hizala
};

void paralel_sayim() {
    const int IS_PARCACIGI_SAYISI = 4;
    IyiSayac sayaclar[IS_PARCACIGI_SAYISI];
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        for(int i = 0; i < 1000000; i++) {
            sayaclar[tid].sayac++;
        }
    }
}
```

---

// ... Hafta-6 için detaylı içerik devam edecek 