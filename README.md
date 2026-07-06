# KTP Forensic & Fraud Detector

Aplikasi *Classical Computer Vision* untuk mendeteksi manipulasi, editan, dan kepalsuan pada citra KTP Indonesia tanpa menggunakan model *Deep Learning* yang berat. Seluruh pemrosesan berjalan secara lokal dengan kecepatan tinggi (< 5 detik per citra).

## Fitur Modul (12+ Analisis)
1. Image Quality Analysis (Blur, Exposure, Resolution)
2. Metadata (EXIF) & Suspicious Software Detection
3. OCR Data Extraction (14 Kolom KTP)
4. Logical Validation (NIK vs Tanggal Lahir vs Gender)
5. Error Level Analysis (ELA)
6. Frequency Domain Analysis (FFT Spectrum)
7. Noise Consistency & Block Analysis
8. Color Space & Photocopy/Screen Detection
9. Edge Density & Anomaly Detection
10. Copy-Move Forgery Detection (ORB Keypoints)
11. Text Region & Baseline Alignment Analysis
12. Risk Scoring Engine (0-100)

## Prasyarat Sistem
Aplikasi ini membutuhkan Tesseract OCR Engine terinstal di OS Anda:
- **Windows**: Download dan install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki). Pastikan Language Pack **Indonesian (ind)** ikut terinstal.
- **Linux / Ubuntu / Google Colab**:
  ```bash
  sudo apt-get update
  sudo apt-get install tesseract-ocr tesseract-ocr-ind