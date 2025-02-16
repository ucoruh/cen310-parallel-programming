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
header: 'CEN310 Paralel Programlama Hafta-15'
footer: '![height:50px](http://erdogan.edu.tr/Images/Uploads/MyContents/L_379-20170718142719217230.jpg) RTEÜ CEN310 Hafta-15'
title: "CEN310 Paralel Programlama Hafta-15"
author: "Öğr. Gör. Dr. Uğur CORUH"
date:
subtitle: "Final Proje Değerlendirmesi"
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

## Hafta-15 (Final Proje Değerlendirmesi)

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

## Final Proje Gereksinimleri

### 1. Proje Dokümantasyonu
- Kapsamlı proje raporu
- Kaynak kod dokümantasyonu
- Performans analizi sonuçları
- Uygulama detayları
- Gelecek çalışma önerileri

### 2. Teknik Uygulama
- Çalışan paralel uygulama
- Çoklu paralel programlama modelleri
- İleri optimizasyon teknikleri
- Hata yönetimi ve sağlamlık
- Kod kalitesi ve organizasyonu

---

## Sunum Yönergeleri

### Format
- Grup başına 30 dakika
- 20 dakika sunum
- 10 dakika soru-cevap

### İçerik
1. Proje Genel Bakışı
   - Problem tanımı
   - Çözüm yaklaşımı
   - Teknik zorluklar

2. Uygulama Detayları
   - Mimari tasarım
   - Paralel stratejiler
   - Optimizasyon teknikleri

3. Sonuçlar ve Analiz
   - Performans ölçümleri
   - Ölçeklenebilirlik testleri
   - Karşılaştırmalı analiz

4. Canlı Demo
   - Sistem kurulumu
   - Özellik gösterimi
   - Performans sunumu

---

## Performans Analizi Gereksinimleri

### Ölçülecek Metrikler
- Çalışma süresi
- Hızlanma
- Verimlilik
- Kaynak kullanımı
- Ölçeklenebilirlik

### Analiz Araçları
```bash
# Performans ölçüm örnekleri
$ nvprof ./cuda_programi
$ mpirun -np 4 ./mpi_programi
$ perf stat ./openmp_programi
```

---

## Proje Yapısı Örneği

```cpp
proje/
├── src/
│   ├── main.cpp
│   ├── cuda/
│   │   ├── cekirdek.cu
│   │   └── gpu_yardimcilar.cuh
│   ├── mpi/
│   │   ├── iletisimci.cpp
│   │   └── veri_transfer.h
│   └── openmp/
│       ├── paralel_donguler.cpp
│       └── is_parcacigi_yardimcilar.h
├── include/
│   ├── ortak.h
│   └── yapilandirma.h
├── test/
│   ├── birim_testler.cpp
│   └── performans_testleri.cpp
├── docs/
│   ├── rapor.pdf
│   └── sunum.pptx
├── veri/
│   ├── girdi/
│   └── cikti/
├── betikler/
│   ├── derle.sh
│   └── testleri_calistir.sh
├── CMakeLists.txt
└── README.md
```

---

## Değerlendirme Kriterleri

### Teknik Yönler (50%)
- Uygulama kalitesi (15%)
- Performans optimizasyonu (15%)
- Kod organizasyonu (10%)
- Hata yönetimi (10%)

### Dokümantasyon (25%)
- Proje raporu (10%)
- Kod dokümantasyonu (10%)
- Sunum kalitesi (5%)

### Sonuçlar ve Analiz (25%)
- Performans sonuçları (10%)
- Karşılaştırmalı analiz (10%)
- Gelecek iyileştirmeler (5%)

---

## Yaygın Proje Konuları

1. Bilimsel Hesaplama
   - N-cisim simülasyonları
   - Akışkanlar dinamiği
   - Monte Carlo yöntemleri
   - Matris hesaplamaları

2. Veri İşleme
   - Görüntü/video işleme
   - Sinyal işleme
   - Veri madenciliği
   - Örüntü tanıma

3. Makine Öğrenmesi
   - Sinir ağı eğitimi
   - Paralel model çıkarımı
   - Veri ön işleme
   - Özellik çıkarımı

4. Graf İşleme
   - Yol bulma
   - Graf analitiği
   - Ağ analizi
   - Ağaç algoritmaları

---

## Kaynaklar ve Referanslar

### Dokümantasyon
- CUDA Programlama Kılavuzu
- OpenMP API Spesifikasyonu
- MPI Standart Dokümantasyonu
- Performans Optimizasyon Kılavuzları

### Araçlar
- Visual Studio
- NVIDIA NSight
- Intel VTune
- Performans Profilleyiciler

---

## Proje Raporu Şablonu

### 1. Giriş
- Arka plan
- Hedefler
- Kapsam

### 2. Tasarım
- Sistem mimarisi
- Bileşen tasarımı
- Paralel stratejiler

### 3. Uygulama
- Geliştirme ortamı
- Teknik detaylar
- Optimizasyon teknikleri

### 4. Sonuçlar
- Performans ölçümleri
- Analiz
- Karşılaştırmalar

### 5. Sonuç
- Başarılar
- Zorluklar
- Gelecek çalışmalar

---

## İletişim Bilgileri

Proje ile ilgili sorularınız için:

- **E-posta:** ugur.coruh@erdogan.edu.tr
- **Ofis Saatleri:** Randevu ile
- **Konum:** Mühendislik Fakültesi

---

<!-- _backgroundColor: aquq -->

<!-- _color: orange -->

# Sorular ve Tartışma

--- 