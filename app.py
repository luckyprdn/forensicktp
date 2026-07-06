"""
KTP Forensic & Fraud Detector App
A single-file Streamlit application using Fast OpenCV Engine & Fuzzy Logic OCR.
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
# scipy.stats dihapus karena kita memakai native NumPy entropy dari engine Anda

# ==========================================
# CONFIGURATION
# ==========================================
# Hapus tanda komentar di bawah ini jika menjalankan di Windows Lokal
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class KTPForensicAnalyzer:
    def __init__(self):
        # --- Thresholds Configuration ---
        self.THRESH_BLUR = 40.0
        self.THRESH_SATURATION = 45.0
        self.THRESH_ENTROPY = 6.0 
        self.THRESH_ELA_TOP_PERCENTILE = 40.0
        self.THRESH_NOISE_RATIO = 0.40 
        self.THRESH_FFT_ENERGY = 180.0
        
        self.SUSPICIOUS_SOFTWARE = ['photoshop', 'canva', 'gimp', 'illustrator', 'lightroom', 'midjourney', 'stable diffusion', 'dall-e']
        
        self.TESSERACT_CONFIG = r'-l ind --oem 3 --psm 6'

    def analyze(self, image_bytes: bytes) -> dict:
        start_time = time.time()
        
        try:
            # 1. Load Image from Bytes (Streamlit Upload)
            nparr = np.frombuffer(image_bytes, np.uint8)
            cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if cv_img is None:
                raise ValueError("Gambar tidak valid atau rusak.")
            
            # Pre-calculate color spaces
            cv_gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            cv_hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
            
            # 2. Run Forensic Modules (Fast OpenCV / Numpy)
            quality_metrics = self._analyze_quality(cv_gray)
            meta_risk, meta_details = self._analyze_metadata(image_bytes)
            copy_risk, copy_details = self._analyze_photocopy(cv_hsv, cv_gray)
            
            # Modul dengan visualisasi untuk UI
            ela_risk, ela_details, ela_vis = self._analyze_ela_cv2(cv_img) 
            freq_risk, freq_details, freq_vis = self._analyze_frequency_cv2(cv_gray) 
            noise_risk, noise_details, noise_vis = self._analyze_noise_fast(cv_gray) 
            ocr_risk, ocr_details, ocr_vis = self._analyze_ocr_logic(cv_gray, cv_img)
            
            # 3. Risk Engine
            final_score, judgment, flags = self._risk_engine(
                quality_metrics, meta_risk, copy_risk, ela_risk, freq_risk, noise_risk, ocr_risk,
                meta_details, ocr_details
            )
            
            return {
                "status": "SUCCESS",
                "final_judgment": judgment,
                "confidence_score": float(round(final_score, 2)),
                "triggered_flags": flags,
                "processing_time_ms": float(round((time.time() - start_time) * 1000, 2)),
                "forensic_metrics": {
                    "image_quality": quality_metrics,
                    "metadata": meta_details,
                    "photocopy_analysis": copy_details,
                    "compression_ela": ela_details,
                    "frequency_analysis": freq_details,
                    "texture_noise": noise_details,
                    "ocr_validation": ocr_details
                },
                "visualizations": {
                    "ELA_Heatmap": ela_vis,
                    "FFT_Spectrum": freq_vis,
                    "Noise_Map": noise_vis,
                    "OCR_Image": ocr_vis
                }
            }

        except Exception as e:
            return {"status": "ERROR", "message": str(e)}

    # ==========================================
    # MODUL 1: IMAGE QUALITY
    # ==========================================
    def _analyze_quality(self, cv_gray) -> dict:
        laplacian_var = cv2.Laplacian(cv_gray, cv2.CV_64F).var()
        brightness = np.mean(cv_gray)
        contrast = np.std(cv_gray)
        return {
            "is_poor_quality": bool(laplacian_var < self.THRESH_BLUR or brightness > 210 or brightness < 45),
            "laplacian_variance": float(round(laplacian_var, 2)),
            "brightness": float(round(brightness, 2)),
            "contrast": float(round(contrast, 2))
        }

    # ==========================================
    # MODUL 2: DEEP EXIF FORENSICS (Bytes Input)
    # ==========================================
    def _analyze_metadata(self, image_bytes: bytes):
        tags = exifread.process_file(io.BytesIO(image_bytes), details=False)
            
        if not tags: 
            return False, {"status": "No EXIF (Cleaned / Social Media Download)"}
            
        details = {}
        risk = False
        
        for tag_name, tag_value in tags.items():
            val_str = str(tag_value).lower()
            if 'Software' in tag_name or 'Processing' in tag_name:
                details['software'] = val_str
                if any(kw in val_str for kw in self.SUSPICIOUS_SOFTWARE): risk = True
            elif 'DateTime' in tag_name:
                details[tag_name] = val_str
                
        if 'Image DateTime' in details and 'EXIF DateTimeOriginal' in details:
            if details['Image DateTime'] != details['EXIF DateTimeOriginal']:
                risk = True; details['date_anomaly'] = "Waktu modifikasi tidak sesuai waktu asli"
                
        return risk, details

    # ==========================================
    # MODUL 3: PHOTOCOPY (NATIVE NUMPY ENTROPY)
    # ==========================================
    def _analyze_photocopy(self, cv_hsv, cv_gray):
        sat_channel = cv_hsv[:, :, 1]
        sat_mean = np.mean(sat_channel)
        
        hist = cv2.calcHist([cv_gray], [0], None, [256], [0, 256]).ravel()
        hist = hist[hist > 0] 
        hist = hist / (hist.sum() + 1e-7)
        img_entropy = -np.sum(hist * np.log2(hist))
        
        risk = (sat_mean < self.THRESH_SATURATION) or (sat_mean < 50 and img_entropy < self.THRESH_ENTROPY)
        return risk, {"saturation_mean": float(round(sat_mean, 2)), "shannon_entropy": float(round(img_entropy, 2))}

    # ==========================================
    # MODUL 4: ERROR LEVEL ANALYSIS (FAST CV2)
    # ==========================================
    def _analyze_ela_cv2(self, cv_img):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, encimg = cv2.imencode('.jpg', cv_img, encode_param)
        resaved = cv2.imdecode(encimg, 1)
        
        diff = cv2.absdiff(cv_img, resaved)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        top_1_percent = np.percentile(diff_gray, 99)
        mean_of_top = np.mean(diff_gray[diff_gray >= top_1_percent])
        
        # Visualisasi untuk Streamlit
        ela_vis = cv2.applyColorMap(cv2.convertScaleAbs(diff_gray, alpha=15.0), cv2.COLORMAP_JET)
        
        return mean_of_top > self.THRESH_ELA_TOP_PERCENTILE, {"ela_top_1_percentile_mean": float(round(mean_of_top, 2))}, ela_vis

    # ==========================================
    # MODUL 5: FREQUENCY ANALYSIS (OPENCV DFT)
    # ==========================================
    def _analyze_frequency_cv2(self, cv_gray):
        rows, cols = cv_gray.shape
        opt_rows = cv2.getOptimalDFTSize(rows)
        opt_cols = cv2.getOptimalDFTSize(cols)
        padded = cv2.copyMakeBorder(cv_gray, 0, opt_rows - rows, 0, opt_cols - cols, cv2.BORDER_CONSTANT, value=[0])
        
        dft = cv2.dft(np.float32(padded), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
        
        crow, ccol = opt_rows // 2, opt_cols // 2
        mask = np.ones((opt_rows, opt_cols), np.uint8)
        cv2.circle(mask, (ccol, crow), 60, 0, -1)
        
        high_freq = magnitude * mask
        high_freq_mean = np.mean(high_freq[high_freq > 0])
        
        # Visualisasi untuk Streamlit
        mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        freq_vis = cv2.applyColorMap(mag_norm, cv2.COLORMAP_MAGMA)
        
        return high_freq_mean > self.THRESH_FFT_ENERGY, {"high_freq_energy_mean": float(round(high_freq_mean, 2))}, freq_vis

    # ==========================================
    # MODUL 6: NOISE CONSISTENCY 
    # ==========================================
    def _analyze_noise_fast(self, cv_gray):
        h, w = cv_gray.shape
        bh, bw = h // 4, w // 4
        variances = []
        noise_map = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(4):
            for j in range(4):
                block = cv_gray[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
                var = cv2.Laplacian(block, cv2.CV_64F).var()
                variances.append(var)
                noise_map[i*bh:(i+1)*bh, j*bw:(j+1)*bw] = int(min(var, 255))
                
        valid_variances = [v for v in variances if v > 5.0]
        noise_vis = cv2.applyColorMap(noise_map, cv2.COLORMAP_VIRIDIS)
        
        if len(valid_variances) < 4:
            return False, {"noise_ratio": 1.0, "note": "Blok terlalu solid/hitam"}, noise_vis
            
        min_var = np.percentile(valid_variances, 10)
        max_var = np.percentile(valid_variances, 90)
        
        ratio = min_var / max_var if max_var > 0 else 1.0
        risk = ratio < self.THRESH_NOISE_RATIO
        
        return risk, {"noise_consistency_ratio": float(round(ratio, 2))}, noise_vis

    # ==========================================
    # MODUL 7: FUZZY LOGIC OCR
    # ==========================================
    def _analyze_ocr_logic(self, cv_gray, cv_img):
        h, w = cv_gray.shape
        scale_ratio = 1.0
        if w > 1000:
            scale_ratio = 1000 / w
            cv_gray = cv2.resize(cv_gray, (1000, int(h * scale_ratio)), interpolation=cv2.INTER_LINEAR)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast = clahe.apply(cv_gray)
        thresh = cv2.adaptiveThreshold(contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
        
        text = pytesseract.image_to_string(thresh, config=self.TESSERACT_CONFIG)
        text_upper = text.upper()
        
        nik_match = re.search(r'\b\d{16}\b', text)
        dob_match = re.search(r'\b(\d{2})-(\d{2})-(\d{4})\b', text)
        sex_match = re.search(r'\b(LAKI-LAKI|PEREMPUAN)\b', text_upper)
        
        details = {
            "nik_extracted": nik_match.group(0) if nik_match else None,
            "dob_extracted": dob_match.group(0) if dob_match else None,
            "sex_extracted": sex_match.group(0) if sex_match else None,
            "raw_text": text,
            "logic_fail": False,
            "logic_notes": []
        }
        
        risk = False
        
        if details["nik_extracted"] and details["dob_extracted"]:
            nik = details["nik_extracted"]
            nik_dd, nik_mm, nik_yy = int(nik[6:8]), int(nik[8:10]), int(nik[10:12])
            dob_dd, dob_mm, dob_yy = int(dob_match.group(1)), int(dob_match.group(2)), int(dob_match.group(3)[2:4])
            
            is_female_nik = nik_dd > 40
            actual_dd = nik_dd - 40 if is_female_nik else nik_dd
            
            nik_date_str = f"{actual_dd:02d}{nik_mm:02d}{nik_yy:02d}"
            dob_date_str = f"{dob_dd:02d}{dob_mm:02d}{dob_yy:02d}"
            
            diff_count = sum(1 for a, b in zip(nik_date_str.ljust(6), dob_date_str.ljust(6)) if a != b)
            
            if diff_count > 1:
                if 1 <= dob_mm <= 12 and 1 <= dob_dd <= 31:
                    risk = True
                    details["logic_fail"] = True
                    details["logic_notes"].append(f"Logic Mismatch (Edit): NIK ({nik_date_str}) vs DOB ({dob_date_str})")
                else:
                    details["logic_notes"].append(f"OCR Error/Typo diabaikan: NIK({nik_date_str}) vs DOB({dob_date_str})")
            elif diff_count == 1:
                 details["logic_notes"].append("OCR typo 1 angka terdeteksi (Diampuni).")

            if details["sex_extracted"]:
                expected_sex = "PEREMPUAN" if is_female_nik else "LAKI-LAKI"
                if details["sex_extracted"] != expected_sex:
                    risk = True
                    details["logic_fail"] = True
                    details["logic_notes"].append(f"Logic Mismatch: NIK Gender vs Teks ({details['sex_extracted']})")
                    
        # Visualisasi untuk Streamlit
        vis_img = cv_img.copy()
        try:
            d = pytesseract.image_to_data(thresh, config=self.TESSERACT_CONFIG, output_type=pytesseract.Output.DICT)
            for i in range(len(d['text'])):
                if int(d['conf'][i]) > 60 and len(d['text'][i].strip()) > 2:
                    # Kembalikan koordinat ke skala asli gambar
                    x = int(d['left'][i] / scale_ratio)
                    y = int(d['top'][i] / scale_ratio)
                    w = int(d['width'][i] / scale_ratio)
                    h_box = int(d['height'][i] / scale_ratio)
                    cv2.rectangle(vis_img, (x, y), (x + w, y + h_box), (0, 255, 0), 2)
        except:
            pass
            
        return risk, details, cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)

    # ==========================================
    # MODUL 8: RISK ENGINE (WEIGHTED SCORE)
    # ==========================================
    def _risk_engine(self, quality, meta, copy, ela, freq, noise, ocr, meta_details, ocr_details):
        score = 0.0
        flags = []
        
        if meta:
            score += 45
            flags.append(f"Metadata Forgery: {meta_details.get('software', 'Date Anomaly')}")
        if ocr:
            score += 45
            flags.append(f"Logic Mismatch: {ocr_details.get('logic_notes', ['-'])[0]}")
        if copy:
            score += 40
            flags.append("Photocopy/Screen (Low Color/Entropy)")
            
        multiplier = 0.5 if quality["is_poor_quality"] else 1.0
        
        if ela:
            score += (25 * multiplier)
            flags.append("Compression Anomaly (ELA)")
        if noise:
            score += (20 * multiplier)
            flags.append("Texture Inconsistency (Tempelan OpenCV)")
        if freq:
            score += (15 * multiplier)
            flags.append("Frequency Anomaly (AI/Print Pattern)")
            
        if quality["is_poor_quality"]:
            flags.append("Image Quality Poor (Blur/Exposure)")
            if score < 20: score += 15 
            
        final_score = min(score, 100.0)
        
        if final_score >= 70: judgment = "PALSU / MANIPULASI"
        elif final_score >= 35: judgment = "SUSPECT (Manual Review)"
        else: judgment = "ASLI"
            
        return final_score, judgment, flags


# ==========================================
# STREAMLIT UI APPLICATION
# ==========================================
def main():
    st.set_page_config(page_title="KTP Forensic Detector", layout="wide", page_icon="🛡️")
    
    st.markdown("""
        <style>
        .stProgress > div > div > div > div { background-color: #ff4b4b; }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🛡️ KTP Image Forensic & Fraud Detector")
    st.markdown("Sistem *Classical Computer Vision* yang dioptimasi untuk kecepatan tinggi.")
    
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
            with st.spinner('Menjalankan Modul Forensik...'):
                result = engine.analyze(image_bytes)
                
            if result.get("status") == "ERROR":
                st.error(f"Gagal memproses gambar: {result.get('message')}")
                return
                
            score = result["confidence_score"]
            judgment = result["final_judgment"]
            
            st.markdown(f"### Score Kepalsuan: {score}/100")
            st.progress(int(score))
            
            if "PALSU" in judgment:
                st.error(f"**VONIS: {judgment}** (Sangat Mencurigakan)")
            elif "SUSPECT" in judgment:
                st.warning(f"**VONIS: {judgment}** (Perlu Reviu Manual)")
            else:
                st.success(f"**VONIS: {judgment}** (Cenderung Asli)")
                
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
        
        tab_ocr, tab_vis, tab_raw, tab_exif = st.tabs([
            "📝 Fuzzy Logic OCR", 
            "👁️ Visualisasi Forensik", 
            "📊 Raw Metrics", 
            "📸 Metadata EXIF"
        ])
        
        with tab_ocr:
            ocr_data = result["forensic_metrics"]["ocr_validation"]
            
            st.markdown("#### Logika Validasi")
            if ocr_data["logic_notes"]:
                for note in ocr_data["logic_notes"]:
                    st.warning(f"⚠️ {note}")
            else:
                st.success("✅ Logika tanggal & NIK sesuai.")
                
            st.markdown("#### Teks Mentah (Raw Text)")
            st.text_area("", value=ocr_data.get("raw_text", ""), height=150)
                
        with tab_vis:
            vis = result["visualizations"]
            v_col1, v_col2 = st.columns(2)
            with v_col1:
                st.image(vis["ELA_Heatmap"], caption="Error Level Analysis (Heatmap Kompresi)", use_column_width=True)
                st.image(vis["Noise_Map"], caption="Noise Consistency (Deteksi Tempelan)", use_column_width=True)
            with v_col2:
                st.image(vis["FFT_Spectrum"], caption="FFT Frequency Spectrum (Deteksi Print/AI)", use_column_width=True)
                st.image(vis["OCR_Image"], caption="OCR Bounding Boxes", use_column_width=True)
                
        with tab_raw:
            st.json({k: v for k, v in result["forensic_metrics"].items() if k != "metadata"})
            
        with tab_exif:
            meta = result["forensic_metrics"]["metadata"]
            st.write(f"**Status:** {meta.get('status', 'Analyzed')}")
            st.json(meta)

if __name__ == "__main__":
    main()