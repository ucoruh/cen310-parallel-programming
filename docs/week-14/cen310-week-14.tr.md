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
header: 'CEN310 Paralel Programlama Hafta-14'
footer: '![height:50px](http://erdogan.edu.tr/Images/Uploads/MyContents/L_379-20170718142719217230.jpg) RTEÜ CEN310 Hafta-14'
title: "CEN310 Paralel Programlama Hafta-14"
author: "Öğr. Gör. Dr. Uğur CORUH"
date:
subtitle: "Quiz-2"
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

## Hafta-14 (Quiz-2)

#### Bahar Dönemi, 2024-2025

---

## Quiz-2 Bilgileri

### Tarih ve Saat
- **Tarih:** 16 Mayıs 2025
- **Saat:** 09:00-12:00 (3 saat)
- **Konum:** Normal sınıf

### Format
- Yazılı sınav
- Teorik ve pratik soruların karışımı
- Hem kapalı hem açık uçlu sorular

---

## Kapsanan Konular

### 1. GPU Programlama
- CUDA Mimarisi
- Bellek Hiyerarşisi
- İş Parçacığı Organizasyonu
- Performans Optimizasyonu

### 2. İleri Paralel Desenler
- Boru Hattı İşleme
- Görev Paralelliği
- Veri Paralelliği
- Hibrit Yaklaşımlar

### 3. Gerçek Dünya Uygulamaları
- Bilimsel Hesaplama
- Veri İşleme
- Matris İşlemleri
- N-cisim Simülasyonları

---

## Örnek Sorular

### Teorik Sorular
1. CUDA bellek hiyerarşisini ve performansa etkisini açıklayın.
2. Farklı paralel desenleri ve kullanım durumlarını karşılaştırın.
3. GPU programları için optimizasyon stratejilerini tanımlayın.

### Pratik Problemler
```cpp
// Soru 1: Bu CUDA programının çıktısı nedir?
__global__ void cekirdek(int* veri) {
    int idx = threadIdx.x;
    __shared__ int paylasimli_veri[256];
    
    paylasimli_veri[idx] = veri[idx];
    __syncthreads();
    
    if(idx < 128) {
        paylasimli_veri[idx] += paylasimli_veri[idx + 128];
    }
    __syncthreads();
    
    if(idx == 0) {
        veri[0] = paylasimli_veri[0];
    }
}

int main() {
    int* veri;
    // ... başlatma kodu ...
    cekirdek<<<1, 256>>>(veri);
    // ... temizleme kodu ...
}
```

---

## Hazırlık Yönergeleri

### 1. İncelenecek Materyaller
- Ders slaytları ve notları
- Laboratuvar alıştırmaları
- Örnek kodlar
- Pratik problemler

### 2. Odak Alanları
- CUDA Programlama
- Bellek Yönetimi
- Performans Optimizasyonu
- Gerçek Dünya Uygulamaları

### 3. Pratik Alıştırmalar
- CUDA programları yazma ve analiz etme
- Paralel desenleri uygulama
- Mevcut kodu optimize etme
- Performans ölçümü

---

## Sınav Kuralları

1. **İzin Verilen Materyaller**
   - Kitap veya not kullanımı yasak
   - Elektronik cihaz kullanımı yasak
   - Müsvedde için temiz kağıt

2. **Zaman Yönetimi**
   - Tüm soruları dikkatlice okuyun
   - Her bölüm için zamanınızı planlayın
   - İnceleme için zaman bırakın

3. **Soruları Yanıtlama**
   - Tüm çalışmanızı gösterin
   - Mantığınızı açıklayın
   - Açık ve düzenli yazın

---

## Değerlendirme Kriterleri

### Dağılım
- Teorik Sorular: 40%
- Pratik Problemler: 60%

### Değerlendirme
- Kavramları anlama
- Problem çözme yaklaşımı
- Kod analizi ve yazımı
- Performans değerlendirmeleri
- Açık açıklamalar

---

## Ek Kaynaklar

### İnceleme Materyalleri
- CUDA Programlama Kılavuzu
- Performans Optimizasyon Kılavuzu
- Örnek Uygulamalar
- Çevrimiçi Dokümantasyon:
  - [CUDA Dokümantasyonu](https://docs.nvidia.com/cuda/)
  - [OpenMP Referansı](https://www.openmp.org/)
  - [MPI Dokümantasyonu](https://www.open-mpi.org/)

### Örnek Kod Deposu
- Ders GitHub deposu
- Örnek uygulamalar
- Performans kıyaslamaları

---

## İletişim Bilgileri

Sınav ile ilgili sorularınız için:

- **E-posta:** ugur.coruh@erdogan.edu.tr
- **Ofis Saatleri:** Randevu ile
- **Konum:** Mühendislik Fakültesi

---

<!-- _backgroundColor: aquq -->

<!-- _color: orange -->

# Başarılar!

--- 