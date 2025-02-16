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
header: 'CEN310 Paralel Programlama Hafta-8'
footer: '![height:50px](http://erdogan.edu.tr/Images/Uploads/MyContents/L_379-20170718142719217230.jpg) RTEÜ CEN310 Hafta-8'
title: "CEN310 Paralel Programlama Hafta-8"
author: "Öğr. Gör. Dr. Uğur CORUH"
date:
subtitle: "Vize Proje Değerlendirmesi"
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

## Hafta-8 (Vize Proje Değerlendirmesi)

#### Bahar Dönemi, 2024-2025

---

## Proje Değerlendirme Günü Programı

### Sabah Oturumu (09:00-12:00)
- Proje sunumları (Grup 1-4)
- Performans analizi tartışmaları
- Soru-cevap oturumları

### Öğle Arası (12:00-13:00)

### Öğleden Sonra Oturumu (13:00-17:00)
- Proje sunumları (Grup 5-8)
- Teknik gösterimler
- Son değerlendirmeler

---

## Proje Gereksinimleri

### 1. Dokümantasyon
- Proje raporu
- Kaynak kod dokümantasyonu
- Performans analizi sonuçları
- Uygulama zorlukları
- Gelecek iyileştirmeler

### 2. Uygulama
- Çalışan paralel program
- OpenMP ve/veya MPI kullanımı
- Performans optimizasyonları
- Hata yönetimi
- Kod kalitesi

---

## Sunum Yönergeleri

### Format
- Grup başına 20 dakika
- 15 dakika sunum
- 5 dakika soru-cevap

### İçerik
1. Problem Tanımı
2. Çözüm Yaklaşımı
3. Uygulama Detayları
4. Performans Sonuçları
5. Zorluklar ve Çözümler
6. Demo

---

## Performans Analizi Gereksinimleri

### Ölçülecek Metrikler
- Çalışma süresi
- Hızlanma
- Verimlilik
- Ölçeklenebilirlik
- Kaynak kullanımı

### Analiz Araçları
```bash
# Örnek performans ölçümü
$ perf stat ./paralel_program
$ nvprof ./cuda_program
$ vtune ./openmp_program
```

---

## Örnek Proje Yapısı

```cpp
// Proje mimarisi örneği
proje/
├── src/
│   ├── main.cpp
│   ├── paralel_uygulama.cpp
│   └── yardimcilar.cpp
├── include/
│   ├── paralel_uygulama.h
│   └── yardimcilar.h
├── testler/
│   └── test_paralel.cpp
├── dokumanlar/
│   ├── rapor.pdf
│   └── sunum.pptx
└── README.md
```

---

## Performans Karşılaştırma Şablonu

### Sıralı vs Paralel Uygulama

```cpp
// Sıralı uygulama
double sirali_sure = 0.0;
{
    auto baslangic = std::chrono::high_resolution_clock::now();
    sirali_sonuc = sirali_hesapla();
    auto bitis = std::chrono::high_resolution_clock::now();
    sirali_sure = std::chrono::duration<double>(bitis-baslangic).count();
}

// Paralel uygulama
double paralel_sure = 0.0;
{
    auto baslangic = std::chrono::high_resolution_clock::now();
    paralel_sonuc = paralel_hesapla();
    auto bitis = std::chrono::high_resolution_clock::now();
    paralel_sure = std::chrono::duration<double>(bitis-baslangic).count();
}

// Hızlanma hesapla
double hizlanma = sirali_sure / paralel_sure;
```

---

## Yaygın Proje Konuları

1. Matris İşlemleri
   - Matris çarpımı
   - Matris ayrıştırma
   - Lineer denklem çözümü

2. Bilimsel Hesaplama
   - N-cisim simülasyonu
   - Dalga denklemi çözücü
   - Monte Carlo yöntemleri

3. Veri İşleme
   - Görüntü işleme
   - Sinyal işleme
   - Veri madenciliği

4. Graf Algoritmaları
   - En kısa yol
   - Graf boyama
   - Maksimum akış

---

## Değerlendirme Kriterleri

### Teknik Yönler (60%)
- Doğru uygulama (20%)
- Performans optimizasyonu (20%)
- Kod kalitesi (10%)
- Dokümantasyon (10%)

### Sunum (40%)
- Açık anlatım (15%)
- Demo kalitesi (15%)
- Soru-cevap yönetimi (10%)

---

## Proje Raporu Şablonu

### 1. Giriş
- Problem tanımı
- Hedefler
- Arka plan

### 2. Tasarım
- Mimari
- Algoritmalar
- Paralelleştirme stratejisi

### 3. Uygulama
- Kullanılan teknolojiler
- Kod yapısı
- Temel bileşenler

### 4. Sonuçlar
- Performans ölçümleri
- Analiz
- Karşılaştırmalar

### 5. Sonuç
- Başarılar
- Zorluklar
- Gelecek çalışmalar

---

## Kaynaklar ve Referanslar

### Dokümantasyon
- OpenMP: [https://www.openmp.org/](https://www.openmp.org/)
- MPI: [https://www.open-mpi.org/](https://www.open-mpi.org/)
- CUDA: [https://docs.nvidia.com/cuda/](https://docs.nvidia.com/cuda/)

### Araçlar
- Performans analiz araçları
- Hata ayıklama araçları
- Profilleme araçları

---

<!-- _backgroundColor: aquq -->

<!-- _color: orange -->

# Sorular ve Tartışma

--- 