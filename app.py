"""
KTP Forensic & Fraud Detector App
A single-file Streamlit application for analyzing Indonesian ID Cards (KTP)
using Classical Computer Vision and Digital Image Forensics.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageChops
import io
import time
import pytesseract
import re
import exifread
from scipy.stats import entropy
from typing import Tuple, Dict, Any, List

# ==========================================
# CONFIGURATION
# ==========================================
# Sesuaikan path ini jika Anda menggunakan Windows dan Tesseract tidak masuk ke PATH.
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class KTPForensicAnalyzer:
    """
    Core Engine untuk Analisis Forensik Citra KTP.
    Mencakup 11 modul forensik dan 1 risk engine.
    """
    
    def __init__(self):
        # Thresholds
        self.MAX_IMAGE_WIDTH = 1500  # Resizing untuk optimasi performa (< 5 detik)
        self.THRESH_BLUR = 35.0
        self.THRESH_SATURATION = 45.0
        self.THRESH_ENTROPY = 5.5 
        self.THRESH_ELA_PERCENTILE = 35.0
        self.THRESH_NOISE_RATIO = 0.35 
        self.THRESH_FFT_ENERGY = 175.0
        
        self.SUSPICIOUS_SOFTWARE = [
            'photoshop', 'canva', 'gimp', 'illustrator', 'lightroom', 
            'midjourney', 'stable diffusion', 'dall-e', 'picsart', 'snapseed'
        ]
        
        # Tesseract Config: -l ind (Bahasa), --oem 3 (Default), --psm 4 (Assume a single column of text of variable sizes)
        self.TESSERACT_CONFIG = r'-l ind --oem 3 --psm 4'

    def analyze(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Menjalankan seluruh pipeline forensik pada byte citra.
        """
        start_time = time.time()
        
        try:
            # 1. Dekode Citra
            nparr = np.frombuffer(image_bytes, np.uint8)
            cv_img_raw = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if cv_img_raw is None:
                raise ValueError("Format file tidak didukung atau corrupt.")

            # Resize untuk performa yang terprediksi (<5 detik)
            cv_img = self._resize_image(cv_img_raw, width=self.MAX_IMAGE_WIDTH)
            
            # Pre-calculate ruang warna
            cv_gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            cv_hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
            
            # --- EKSEKUSI 11 MODUL FORENSIK ---
            quality_res = self._analyze_quality(cv_img, cv_gray)
            meta_res = self._analyze_metadata(image_bytes)
            color_res = self._analyze_color(cv_hsv, cv_gray)
            ela_res, ela_vis = self._analyze_ela(cv_img) 
            freq_res, freq_vis = self._analyze_frequency(cv_gray) 
            noise_res, noise_vis = self._analyze_noise(cv_gray) 
            edge_res, edge_vis = self._analyze_edges(cv_gray)
            copy_move_res, copy_move_vis = self._analyze_copy_move(cv_gray, cv_img)
            
            # Modul OCR (Text Region, Extraction, Logic Validation)
            ocr_res, text_align_res, ocr_vis = self._analyze_ocr_and_text(cv_gray, cv_img)
            
            # --- RISK ENGINE (Modul 12) ---
            final_score, judgment, flags = self._risk_engine(
                quality_res, meta_res, color_res, ela_res, freq_res, 
                noise_res, edge_res, copy_move_res, ocr_res, text_align_res
            )
            
            exec_time = round((time.time() - start_time) * 1000, 2)
            
            return {
                "status": "SUCCESS",
                "final_judgment": judgment,
                "confidence_score": float(round(final_score, 2)),
                "triggered_flags": flags,
                "processing_time_ms": exec_time,
                "metrics": {
                    "Quality": quality_res,
                    "Metadata": meta_res,
                    "Color_Analysis": color_res,
                    "ELA": ela_res,
                    "Frequency": freq_res,
                    "Noise": noise_res,
                    "Edges": edge_res,
                    "Copy_Move": copy_move_res,
                    "Text_Alignment": text_align_res,
                    "OCR_Validation": ocr_res
                },
                "visualizations": {
                    "Original": cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB),
                    "ELA_Heatmap": ela_vis,
                    "FFT_Spectrum": freq_vis,
                    "Noise_Map": noise_vis,
                    "Edge_Map": edge_vis,
                    "Copy_Move_Map": copy_move_vis,
                    "OCR_Bounding_Boxes": ocr_vis
                }
            }

        except Exception as e:
            return {"status": "ERROR", "message": str(e)}

    # ==========================================
    # INTERNAL MODULES
    # ==========================================

    def _resize_image(self, img: np.ndarray, width: int) -> np.ndarray:
        """Resize citra mempertahankan aspect ratio."""
        h, w = img.shape[:2]
        if w > width:
            ratio = width / float(w)
            dim = (width, int(h * ratio))
            return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return img

    def _analyze_quality(self, cv_img: np.ndarray, cv_gray: np.ndarray) -> Dict[str, Any]:
        """Modul 1: Image Quality Analysis"""
        lap_var = cv2.Laplacian(cv_gray, cv2.CV_64F).var()
        bright = np.mean(cv_gray)
        contrast = np.std(cv_gray)
        h, w = cv_img.shape[:2]
        
        # Estimasi kompresi JPEG sederhana berdasarkan ukuran file encode memory
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        _, enc_100 = cv2.imencode('.jpg', cv_img, encode_param)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, enc_90 = cv2.imencode('.jpg', cv_img, encode_param)
        compression_ratio = len(enc_100) / (len(enc_90) + 1e-5)
        
        is_blurry = lap_var < self.THRESH_BLUR
        is_over = bright > 210
        is_under = bright < 40
        
        return {
            "risk": is_blurry or is_over or is_under,
            "resolution": f"{w}x{h}",
            "laplacian_variance": float(round(lap_var, 2)),
            "brightness": float(round(bright, 2)),
            "contrast": float(round(contrast, 2)),
            "est_jpeg_compression_ratio": float(round(compression_ratio, 2)),
            "is_blurry": bool(is_blurry),
            "is_overexposed": bool(is_over),
            "is_underexposed": bool(is_under)
        }

    def _analyze_metadata(self, image_bytes: bytes) -> Dict[str, Any]:
        """Modul 2: Metadata (EXIF) Analysis"""
        tags = exifread.process_file(io.BytesIO(image_bytes), details=False)
        if not tags:
            return {"risk": False, "status": "No EXIF (Cleaned/Social Media)", "details": []}
            
        details = []
        meta_dict = {}
        risk = False
        
        for tag, val in tags.items():
            if tag not in ('JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote'):
                meta_dict[tag] = str(val)
                val_lower = str(val).lower()
                
                # Deteksi Software
                if 'software' in tag.lower() or 'processing' in tag.lower():
                    if any(kw in val_lower for kw in self.SUSPICIOUS_SOFTWARE):
                        risk = True
                        details.append(f"Suspicious Software: {val}")
                        
        # Anomali DateTime
        dt_img = meta_dict.get('Image DateTime')
        dt_orig = meta_dict.get('EXIF DateTimeOriginal')
        if dt_img and dt_orig and dt_img != dt_orig:
            risk = True
            details.append(f"DateTime Anomaly: Modified on {dt_img}")
            
        return {
            "risk": risk,
            "status": "EXIF Found",
            "details": details,
            "raw_meta": meta_dict
        }

    def _analyze_color(self, cv_hsv: np.ndarray, cv_gray: np.ndarray) -> Dict[str, Any]:
        """Modul 8: Color Analysis (Deteksi Fotokopi / Hitam Putih)"""
        sat_mean = np.mean(cv_hsv[:, :, 1])
        hist = cv2.calcHist([cv_gray], [0], None, [256], [0, 256]).ravel()
        hist = hist[hist > 0]
        hist = hist / hist.sum()
        entropy_val = -np.sum(hist * np.log2(hist))
        
        is_grayscale = sat_mean < self.THRESH_SATURATION
        is_low_entropy = entropy_val < self.THRESH_ENTROPY
        
        risk = is_grayscale or (sat_mean < 50 and is_low_entropy)
        
        return {
            "risk": risk,
            "saturation_mean": float(round(sat_mean, 2)),
            "entropy": float(round(entropy_val, 2)),
            "is_suspect_photocopy": bool(risk)
        }

    def _analyze_ela(self, cv_img: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
        """Modul 5: Error Level Analysis (ELA)"""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, encimg = cv2.imencode('.jpg', cv_img, encode_param)
        resaved = cv2.imdecode(encimg, 1)
        
        diff = cv2.absdiff(cv_img, resaved)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Ekstrak metrik
        ela_mean = np.mean(diff_gray)
        ela_max = np.max(diff_gray)
        top_percentile = np.percentile(diff_gray, 99)
        mean_top = np.mean(diff_gray[diff_gray >= top_percentile])
        
        risk = mean_top > self.THRESH_ELA_PERCENTILE
        
        # Buat visualisasi Heatmap ELA (Enhance contrast untuk visual)
        ela_vis = cv2.applyColorMap(cv2.convertScaleAbs(diff_gray, alpha=15.0), cv2.COLORMAP_JET)
        
        return {
            "risk": bool(risk),
            "ela_mean": float(round(ela_mean, 2)),
            "ela_max": float(ela_max),
            "ela_top_1_percentile_mean": float(round(mean_top, 2))
        }, ela_vis

    def _analyze_frequency(self, cv_gray: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
        """Modul 6: Frequency Domain Analysis (FFT)"""
        rows, cols = cv_gray.shape
        opt_rows, opt_cols = cv2.getOptimalDFTSize(rows), cv2.getOptimalDFTSize(cols)
        padded = cv2.copyMakeBorder(cv_gray, 0, opt_rows - rows, 0, opt_cols - cols, cv2.BORDER_CONSTANT, value=[0])
        
        dft = cv2.dft(np.float32(padded), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
        
        # Hitung energi frekuensi tinggi
        crow, ccol = opt_rows // 2, opt_cols // 2
        mask = np.ones((opt_rows, opt_cols), np.uint8)
        cv2.circle(mask, (ccol, crow), 60, 0, -1) # Block low frequency
        
        high_freq = magnitude * mask
        hf_energy = np.mean(high_freq[high_freq > 0])
        
        risk = hf_energy > self.THRESH_FFT_ENERGY
        
        # Visualisasi
        mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        freq_vis = cv2.applyColorMap(mag_norm, cv2.COLORMAP_MAGMA)
        
        return {
            "risk": bool(risk),
            "high_frequency_energy": float(round(hf_energy, 2))
        }, freq_vis

    def _analyze_noise(self, cv_gray: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
        """Modul 7: Noise Consistency"""
        h, w = cv_gray.shape
        bh, bw = h // 4, w // 4
        variances = []
        
        # Visual map 
        noise_map = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(4):
            for j in range(4):
                block = cv_gray[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
                var = cv2.Laplacian(block, cv2.CV_64F).var()
                variances.append(var)
                # Warnai blok di noise map berdasarkan variance (normalized later)
                noise_map[i*bh:(i+1)*bh, j*bw:(j+1)*bw] = int(min(var, 255))
                
        valid_vars = [v for v in variances if v > 5.0]
        if len(valid_vars) < 4:
            return {"risk": False, "noise_ratio": 1.0, "note": "Too solid"}, noise_map
            
        min_var, max_var = np.percentile(valid_vars, 10), np.percentile(valid_vars, 90)
        ratio = min_var / max_var if max_var > 0 else 1.0
        
        risk = ratio < self.THRESH_NOISE_RATIO
        
        noise_vis = cv2.applyColorMap(noise_map, cv2.COLORMAP_VIRIDIS)
        
        return {
            "risk": bool(risk),
            "noise_consistency_ratio": float(round(ratio, 2)),
            "min_block_variance": float(round(min_var, 2)),
            "max_block_variance": float(round(max_var, 2))
        }, noise_vis

    def _analyze_edges(self, cv_gray: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
        """Modul 9: Edge Analysis (Deteksi Tempelan)"""
        # Sobel & Canny
        blur = cv2.GaussianBlur(cv_gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Jika edge terlampau padat atau sangat kosong, mencurigakan
        risk = edge_density > 0.30 or edge_density < 0.01
        
        # Visualisasi
        edge_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edge_vis[edges > 0] = [0, 255, 0] # Hijau untuk edge
        
        return {
            "risk": bool(risk),
            "edge_density": float(round(edge_density, 4))
        }, edge_vis

    def _analyze_copy_move(self, cv_gray: np.ndarray, cv_img: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
        """Modul 10: Copy-Move Detection (ORB)"""
        # Batasi fitur agar cepat
        orb = cv2.ORB_create(nfeatures=500)
        kp, des = orb.detectAndCompute(cv_gray, None)
        
        vis_img = cv_img.copy()
        risk = False
        duplicate_count = 0
        
        if des is not None and len(des) > 10:
            # Match descriptor dengan dirinya sendiri
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = bf.knnMatch(des, des, k=2)
            
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    # Cek rasio jarak (Lowe's ratio test) & pastikan bukan matching dgn dirinya sendiri
                    if m.distance < 0.7 * n.distance and m.queryIdx != m.trainIdx:
                        pt1 = np.array(kp[m.queryIdx].pt)
                        pt2 = np.array(kp[m.trainIdx].pt)
                        # Pastikan jarak spasial cukup jauh (bukan pixel yang bersebelahan)
                        if np.linalg.norm(pt1 - pt2) > 50:
                            good_matches.append(m)
                            
            duplicate_count = len(good_matches)
            if duplicate_count > 10: # Threshold heuristik
                risk = True
                
            # Gambar garis match
            for m in good_matches:
                pt1 = tuple(map(int, kp[m.queryIdx].pt))
                pt2 = tuple(map(int, kp[m.trainIdx].pt))
                cv2.line(vis_img, pt1, pt2, (255, 0, 0), 2)
                
        return {
            "risk": risk,
            "duplicate_keypoints_found": duplicate_count
        }, cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)

    def _analyze_ocr_and_text(self, cv_gray: np.ndarray, cv_img: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any], np.ndarray]:
        """
        Modul 3, 4 & 11: OCR Data Extraction, Logical Validation, dan Text Region Analysis.
        """
        # Preprocessing OCR
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast = clahe.apply(cv_gray)
        thresh = cv2.adaptiveThreshold(contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
        
        # Ekstraksi Data & Bounding Box
        ocr_data = pytesseract.image_to_data(thresh, config=self.TESSERACT_CONFIG, output_type=pytesseract.Output.DICT)
        raw_text = pytesseract.image_to_string(thresh, config=self.TESSERACT_CONFIG)
        
        # --- Modul 11: Text Region & Baseline Alignment ---
        vis_img = cv_img.copy()
        lines_baseline = {} # Menyimpan y-coordinate untuk teks di baris yang sama
        text_risk = False
        text_anomalies = []
        
        n_boxes = len(ocr_data['text'])
        for i in range(n_boxes):
            if int(ocr_data['conf'][i]) > 60 and len(ocr_data['text'][i].strip()) > 2:
                x, y, w, h = (ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i])
                cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                
                # Cek konsistensi tinggi font (anomali tempelan teks beda ukuran)
                if h > 50 or h < 8: 
                    text_risk = True
                    text_anomalies.append(f"Font size anomaly at word '{ocr_data['text'][i]}'")
                    cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
        text_align_res = {
            "risk": text_risk,
            "anomalies": text_anomalies
        }
        
        # --- Modul 3 & 4: Data Extraction & Logical Validation ---
        extracted = self._extract_ktp_fields(raw_text)
        logic_risk, logic_notes = self._validate_ktp_logic(extracted)
        
        ocr_res = {
            "risk": logic_risk,
            "extracted_data": extracted,
            "validation_notes": logic_notes,
            "raw_text_length": len(raw_text)
        }
        
        return ocr_res, text_align_res, cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)

    def _extract_ktp_fields(self, text: str) -> Dict[str, str]:
        """Modul 3: Ekstraksi 14 Kolom menggunakan Regex Heuristik."""
        lines = text.split('\n')
        full_text = " ".join(lines).upper()
        
        data = {
            "NIK": None, "Nama": None, "Tempat_Lahir": None, "Tanggal_Lahir": None,
            "Jenis_Kelamin": None, "Golongan_Darah": None, "Alamat": None, 
            "RT_RW": None, "Kelurahan": None, "Kecamatan": None, "Agama": None,
            "Status_Perkawinan": None, "Pekerjaan": None, "Kewarganegaraan": None,
            "Berlaku_Hingga": None
        }
        
        # 1. NIK
        nik_match = re.search(r'\b\d{16}\b', full_text)
        if nik_match: data["NIK"] = nik_match.group(0)
        
        # 2. Tanggal Lahir (Format DD-MM-YYYY)
        dob_match = re.search(r'\b(\d{2})-(\d{2})-(\d{4})\b', full_text)
        if dob_match: data["Tanggal_Lahir"] = dob_match.group(0)
        
        # 3. Jenis Kelamin
        if "LAKI-LAKI" in full_text or "LAKI" in full_text: data["Jenis_Kelamin"] = "LAKI-LAKI"
        elif "PEREMPUAN" in full_text: data["Jenis_Kelamin"] = "PEREMPUAN"
        
        # 4. Agama
        agamas = ["ISLAM", "KRISTEN", "KATHOLIK", "KATOLIK", "HINDU", "BUDHA", "BUDDHA", "KONGHUCU"]
        for a in agamas:
            if a in full_text:
                data["Agama"] = a
                break
                
        # 5. Status Perkawinan
        if "BELUM KAWIN" in full_text: data["Status_Perkawinan"] = "BELUM KAWIN"
        elif "KAWIN" in full_text: data["Status_Perkawinan"] = "KAWIN"
        elif "CERAI" in full_text: data["Status_Perkawinan"] = "CERAI"
        
        # 6. Kewarganegaraan
        if "WNI" in full_text: data["Kewarganegaraan"] = "WNI"
        elif "WNA" in full_text: data["Kewarganegaraan"] = "WNA"
        
        # 7. Berlaku Hingga
        if "SEUMUR HIDUP" in full_text: data["Berlaku_Hingga"] = "SEUMUR HIDUP"
        
        # Ekstraksi berbasis baris (Heuristik) untuk nama, alamat, dll
        for i, line in enumerate(lines):
            line_up = line.upper()
            if "NAMA" in line_up and ":" in line_up:
                data["Nama"] = line_up.split(":")[-1].strip()
            if "ALAMAT" in line_up and ":" in line_up:
                data["Alamat"] = line_up.split(":")[-1].strip()
            if "RT/RW" in line_up or "RT/" in line_up:
                match = re.search(r'\d{3}\s*/\s*\d{3}', line_up)
                if match: data["RT_RW"] = match.group(0)
            if "KEL/DESA" in line_up and ":" in line_up:
                data["Kelurahan"] = line_up.split(":")[-1].strip()
            if "KECAMATAN" in line_up and ":" in line_up:
                data["Kecamatan"] = line_up.split(":")[-1].strip()
            if "GOL" in line_up and "DARAH" in line_up:
                for gol in [" A ", " B ", " AB ", " O ", " A-", " B-", " O-", " O+", " A+", " B+"]:
                    if gol in line_up: data["Golongan_Darah"] = gol.strip()
                    
        return data

    def _validate_ktp_logic(self, data: Dict[str, str]) -> Tuple[bool, List[str]]:
        """Modul 4: Logical Validation (Validasi silang kolom)"""
        risk = False
        notes = []
        
        nik = data.get("NIK")
        dob = data.get("Tanggal_Lahir")
        gender = data.get("Jenis_Kelamin")
        
        if not nik:
            notes.append("Format NIK (16 digit) tidak ditemukan.")
            return True, notes
            
        # NIK Validation Rules
        nik_dd = int(nik[6:8])
        nik_mm = int(nik[8:10])
        nik_yy = int(nik[10:12])
        
        is_female_nik = nik_dd > 40
        actual_dd = nik_dd - 40 if is_female_nik else nik_dd
        
        # Sanity check NIK Bulan & Tanggal
        if nik_mm > 12 or nik_mm < 1 or actual_dd > 31 or actual_dd < 1:
            risk = True
            notes.append(f"Format Tanggal NIK tidak valid (Tgl:{actual_dd}, Bln:{nik_mm}).")
            
        # Cross-check dengan Tanggal Lahir
        if dob:
            try:
                dob_dd, dob_mm, dob_yy = map(int, dob.split('-'))
                # Cek kemiripan string untuk mentoleransi OCR typo
                nik_date_str = f"{actual_dd:02d}{nik_mm:02d}{nik_yy:02d}"
                dob_date_str = f"{dob_dd:02d}{dob_mm:02d}{str(dob_yy)[-2:]}"
                
                diff_count = sum(1 for a, b in zip(nik_date_str, dob_date_str) if a != b)
                if diff_count > 1:
                    if 1 <= dob_mm <= 12 and 1 <= dob_dd <= 31:
                        risk = True
                        notes.append(f"Logic Mismatch: NIK ({nik_date_str}) vs Tgl Lahir ({dob_date_str}).")
                    else:
                        notes.append(f"OCR Typo pada Tgl Lahir diabaikan.")
            except:
                pass
                
        # Cross-check Gender
        if gender:
            exp_gender = "PEREMPUAN" if is_female_nik else "LAKI-LAKI"
            if gender != exp_gender:
                risk = True
                notes.append(f"Logic Mismatch: Kode Gender NIK vs Teks ({gender}).")
                
        return risk, notes

    def _risk_engine(self, q, meta, color, ela, freq, noise, edge, cm, ocr, txt) -> Tuple[float, str, List[str]]:
        """Modul 12: Confidence & Scoring Engine"""
        score = 0.0
        flags = []
        
        # Hard Flags (Indikasi manusia/logika) - Bobot Tinggi
        if meta["risk"]: 
            score += 40
            flags.append(f"Metadata Forgery: {meta['details'][0] if meta['details'] else 'Anomaly'}")
        if ocr["risk"]:
            score += 35
            flags.append(f"Logic Mismatch: {ocr['validation_notes'][0]}")
        if color["risk"]:
            score += 35
            flags.append("Grayscale/Photocopy (Low Saturation/Entropy)")
        if cm["risk"]:
            score += 35
            flags.append("Copy-Move Forgery Terdeteksi (Clone Stamp)")
            
        # Soft Flags (Indikator piksel/mesin) - Bobot bergantung kualitas
        multiplier = 0.5 if q["risk"] else 1.0
        
        if ela["risk"]: score += (25 * multiplier); flags.append("Compression Anomaly (ELA Edit)")
        if noise["risk"]: score += (20 * multiplier); flags.append("Texture Inconsistency (Noise)")
        if freq["risk"]: score += (15 * multiplier); flags.append("Frequency Anomaly (AI/Print)")
        if edge["risk"]: score += (10 * multiplier); flags.append("Edge Density Anomaly (Splice)")
        if txt["risk"]: score += (10 * multiplier); flags.append("Text Alignment/Size Anomaly")
        
        if q["risk"]:
            flags.append("Image Quality Poor (Blur/Exposure)")
            if score < 20: score += 15 
            
        final_score = min(score, 100.0)
        
        if final_score >= 70: judgment = "PALSU / MANIPULASI"
        elif final_score >= 35: judgment = "SUSPECT"
        else: judgment = "ASLI"
            
        return final_score, judgment, flags


# ==========================================
# STREAMLIT UI APPLICATION
# ==========================================
def main():
    st.set_page_config(page_title="KTP Forensic Detector", layout="wide", page_icon="🛡️")
    
    # CSS Customization
    st.markdown("""
        <style>
        .main { background-color: #f8f9fa; }
        .stProgress > div > div > div > div { background-color: #ff4b4b; }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🛡️ KTP Image Forensic & Fraud Detector")
    st.markdown("Sistem *Classical Computer Vision* untuk menganalisis keaslian citra KTP (Deteksi 12 Modul).")
    
    @st.cache_resource
    def load_engine():
        return KTPForensicAnalyzer()
        
    engine = load_engine()
    
    uploaded_file = st.file_uploader("Unggah Citra KTP (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        
        col_img, col_res = st.columns([1, 1])
        
        with col_img:
            st.subheader("🖼️ Preview KTP")
            st.image(image_bytes, use_column_width=True)
            
        with col_res:
            st.subheader("⚙️ Hasil Analisis")
            with st.spinner('Menjalankan 12 Modul Forensik...'):
                result = engine.analyze(image_bytes)
                
            if result.get("status") == "ERROR":
                st.error(f"Gagal memproses gambar: {result.get('message')}")
                return
                
            score = result["confidence_score"]
            judgment = result["final_judgment"]
            
            # Gauge & Status
            st.markdown(f"### Score Kepalsuan: {score}/100")
            st.progress(int(score))
            
            if "PALSU" in judgment:
                st.error(f"**VONIS: {judgment}** (Sangat Mencurigakan)")
            elif "SUSPECT" in judgment:
                st.warning(f"**VONIS: {judgment}** (Perlu Reviu Manual)")
            else:
                st.success(f"**VONIS: {judgment}** (Cenderung Asli)")
                
            # Flags
            flags = result["triggered_flags"]
            if flags:
                st.markdown("#### 🚨 Anomali / Flags Terdeteksi:")
                for f in flags:
                    st.error(f"- {f}")
            else:
                st.success("✅ Tidak ada anomali terdeteksi.")
                
            st.caption(f"⚡ Waktu Inferensi: {result['processing_time_ms']} ms")
            
        st.markdown("---")
        st.subheader("🔍 Detail Modul Forensik")
        
        # Tabs for details
        tab_ocr, tab_vis, tab_raw, tab_exif = st.tabs([
            "📝 Data OCR & Validasi", 
            "👁️ Visualisasi Forensik", 
            "📊 Raw Metrics (JSON)", 
            "📸 Metadata EXIF"
        ])
        
        with tab_ocr:
            ocr_data = result["metrics"]["OCR_Validation"]
            st.markdown("#### 1. Data Diekstrak")
            
            # Format dict to table
            extracted = ocr_data["extracted_data"]
            clean_extracted = {k: v for k, v in extracted.items() if v is not None}
            if clean_extracted:
                st.table(clean_extracted)
            else:
                st.info("Tidak ada data KTP terstruktur yang berhasil diekstrak.")
                
            st.markdown("#### 2. Logika Validasi NIK")
            if ocr_data["validation_notes"]:
                for note in ocr_data["validation_notes"]:
                    st.warning(f"⚠️ {note}")
            else:
                st.success("✅ Logika tanggal & NIK sesuai.")
                
        with tab_vis:
            st.markdown("*(Peta Forensik ini membantu mendeteksi area manipulasi piksel/tempelan)*")
            vis = result["visualizations"]
            
            v_col1, v_col2 = st.columns(2)
            with v_col1:
                st.image(vis["ELA_Heatmap"], caption="Error Level Analysis (Heatmap Kompresi)", use_column_width=True)
                st.image(vis["Noise_Map"], caption="Noise Consistency (Deteksi Inpainting)", use_column_width=True)
                st.image(vis["OCR_Bounding_Boxes"], caption="OCR Text Regions & Font Anomalies", use_column_width=True)
            with v_col2:
                st.image(vis["FFT_Spectrum"], caption="FFT Frequency Spectrum (Deteksi Print/AI)", use_column_width=True)
                st.image(vis["Edge_Map"], caption="Canny Edge Density (Deteksi Tempelan)", use_column_width=True)
                st.image(vis["Copy_Move_Map"], caption="ORB Copy-Move (Deteksi Clone Stamp)", use_column_width=True)
                
        with tab_raw:
            st.json({k: v for k, v in result["metrics"].items() if k != "Metadata"})
            
        with tab_exif:
            meta = result["metrics"]["Metadata"]
            if meta["risk"]:
                st.error("🚨 Metadata mencurigakan terdeteksi!")
            st.write(f"**Status:** {meta['status']}")
            st.json(meta.get("raw_meta", {}))

if __name__ == "__main__":
    main()