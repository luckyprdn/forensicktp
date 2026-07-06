import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time
import pytesseract
import re
import exifread
import sys
import platform
import json
from typing import Dict, Any, Optional, Tuple, List

# ==========================================
# KTP FORENSIC ANALYZER (Fast OpenCV Engine)
# ==========================================
class KTPForensicAnalyzer:
    """
    Classical Computer Vision forensic engine for KTP image analysis.
    Uses fast NumPy and OpenCV operations to avoid heavy deep learning dependencies.
    """
    def __init__(self) -> None:
        # --- Thresholds Configuration (can be overridden via session_state) ---
        self.THRESH_BLUR = 40.0
        self.THRESH_SATURATION = 45.0
        self.THRESH_ENTROPY = 6.0
        self.THRESH_ELA_TOP_PERCENTILE = 40.0
        self.THRESH_NOISE_RATIO = 0.40
        self.THRESH_FFT_ENERGY = 180.0

        self.SUSPICIOUS_SOFTWARE = [
            'photoshop', 'canva', 'gimp', 'illustrator', 'lightroom',
            'midjourney', 'stable diffusion', 'dall-e'
        ]
        self.TESSERACT_CONFIG = r'-l ind --oem 3 --psm 6'

    @property
    def thresholds(self) -> Dict[str, float]:
        """Return current threshold values for display."""
        return {
            "Blur (Laplacian)": self.THRESH_BLUR,
            "Saturation Mean": self.THRESH_SATURATION,
            "Entropy": self.THRESH_ENTROPY,
            "ELA Top Percentile": self.THRESH_ELA_TOP_PERCENTILE,
            "Noise Consistency Ratio": self.THRESH_NOISE_RATIO,
            "FFT High Freq Energy": self.THRESH_FFT_ENERGY
        }

    def analyze(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Main forensic pipeline.
        Returns a dictionary with status, judgment, metrics and visualizations.
        """
        start_time = time.time()
        try:
            # 1. Decode image
            nparr = np.frombuffer(image_bytes, np.uint8)
            cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if cv_img is None:
                raise ValueError("Invalid or corrupted image.")

            # Prepare color spaces
            cv_gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            cv_hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)

            # 2. Run forensic modules
            quality_metrics = self._analyze_quality(cv_gray)
            meta_risk, meta_details = self._analyze_metadata(image_bytes)
            copy_risk, copy_details = self._analyze_photocopy(cv_hsv, cv_gray)

            ela_risk, ela_details, ela_vis = self._analyze_ela_cv2(cv_img)
            freq_risk, freq_details, freq_vis = self._analyze_frequency_cv2(cv_gray)
            noise_risk, noise_details, noise_vis = self._analyze_noise_fast(cv_gray)
            ocr_risk, ocr_details, ocr_vis = self._analyze_ocr_logic(cv_gray, cv_img)

            # Additional visualizations for dashboard
            gray_bgr = cv2.cvtColor(cv_gray, cv2.COLOR_GRAY2BGR)
            edges = cv2.Canny(cv_gray, 50, 150)
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            _, thresh_img = cv2.threshold(cv_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh_bgr = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)
            hist_data = cv2.calcHist([cv_gray], [0], None, [256], [0, 256]).flatten().tolist()

            # 3. Risk engine
            final_score, judgment, flags = self._risk_engine(
                quality_metrics, meta_risk, copy_risk, ela_risk, freq_risk, noise_risk, ocr_risk,
                meta_details, ocr_details
            )

            # Convert all visualizations to RGB for Streamlit
            visualizations = {
                "Original": cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB),
                "Grayscale": cv2.cvtColor(gray_bgr, cv2.COLOR_BGR2RGB),
                "Edge": cv2.cvtColor(edges_bgr, cv2.COLOR_BGR2RGB),
                "Threshold": cv2.cvtColor(thresh_bgr, cv2.COLOR_BGR2RGB),
                "ELA_Heatmap": cv2.cvtColor(ela_vis, cv2.COLOR_BGR2RGB),
                "Noise_Map": cv2.cvtColor(noise_vis, cv2.COLOR_BGR2RGB),
                "FFT_Spectrum": cv2.cvtColor(freq_vis, cv2.COLOR_BGR2RGB),
                "OCR_Image": cv2.cvtColor(ocr_vis, cv2.COLOR_BGR2RGB),
                "Histogram_Data": hist_data
            }

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
                "visualizations": visualizations
            }
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}

    # ------------------------------------------------------------
    # MODULE 1: Image Quality
    # ------------------------------------------------------------
    def _analyze_quality(self, cv_gray: np.ndarray) -> Dict[str, Any]:
        laplacian_var = cv2.Laplacian(cv_gray, cv2.CV_64F).var()
        brightness = np.mean(cv_gray)
        contrast = np.std(cv_gray)
        return {
            "is_poor_quality": bool(laplacian_var < self.THRESH_BLUR or brightness > 210 or brightness < 45),
            "laplacian_variance": float(round(laplacian_var, 2)),
            "brightness": float(round(brightness, 2)),
            "contrast": float(round(contrast, 2))
        }

    # ------------------------------------------------------------
    # MODULE 2: Metadata EXIF Forensics
    # ------------------------------------------------------------
    def _analyze_metadata(self, image_bytes: bytes) -> Tuple[bool, Dict[str, Any]]:
        try:
            tags = exifread.process_file(io.BytesIO(image_bytes), details=False)
        except Exception:
            tags = {}
        if not tags:
            return False, {"status": "No EXIF (Cleaned / Social Media Download)"}

        details: Dict[str, Any] = {}
        risk = False
        for tag_name, tag_value in tags.items():
            val_str = str(tag_value).lower()
            if 'Software' in tag_name or 'Processing' in tag_name:
                details['software'] = val_str
                if any(kw in val_str for kw in self.SUSPICIOUS_SOFTWARE):
                    risk = True
            elif 'DateTime' in tag_name:
                details[tag_name] = val_str
        if 'Image DateTime' in details and 'EXIF DateTimeOriginal' in details:
            if details['Image DateTime'] != details['EXIF DateTimeOriginal']:
                risk = True
                details['date_anomaly'] = "Modification time does not match original time"
        return risk, details

    # ------------------------------------------------------------
    # MODULE 3: Photocopy Detection (Native NumPy Entropy)
    # ------------------------------------------------------------
    def _analyze_photocopy(self, cv_hsv: np.ndarray, cv_gray: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        sat_channel = cv_hsv[:, :, 1]
        sat_mean = np.mean(sat_channel)
        hist = cv2.calcHist([cv_gray], [0], None, [256], [0, 256]).ravel()
        hist = hist[hist > 0]
        hist = hist / (hist.sum() + 1e-7)
        img_entropy = -np.sum(hist * np.log2(hist))
        risk = (sat_mean < self.THRESH_SATURATION) or (sat_mean < 50 and img_entropy < self.THRESH_ENTROPY)
        return risk, {"saturation_mean": float(round(sat_mean, 2)), "shannon_entropy": float(round(img_entropy, 2))}

    # ------------------------------------------------------------
    # MODULE 4: Error Level Analysis (Fast JPEG re-compression)
    # ------------------------------------------------------------
    def _analyze_ela_cv2(self, cv_img: np.ndarray) -> Tuple[bool, Dict[str, Any], np.ndarray]:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, encimg = cv2.imencode('.jpg', cv_img, encode_param)
        resaved = cv2.imdecode(encimg, 1)
        diff = cv2.absdiff(cv_img, resaved)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        top_1_percent = np.percentile(diff_gray, 99)
        mean_of_top = np.mean(diff_gray[diff_gray >= top_1_percent])
        ela_vis = cv2.applyColorMap(cv2.convertScaleAbs(diff_gray, alpha=15.0), cv2.COLORMAP_JET)
        return mean_of_top > self.THRESH_ELA_TOP_PERCENTILE, {"ela_top_1_percentile_mean": float(round(mean_of_top, 2))}, ela_vis

    # ------------------------------------------------------------
    # MODULE 5: Frequency Analysis (OpenCV DFT)
    # ------------------------------------------------------------
    def _analyze_frequency_cv2(self, cv_gray: np.ndarray) -> Tuple[bool, Dict[str, Any], np.ndarray]:
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
        mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        freq_vis = cv2.applyColorMap(mag_norm, cv2.COLORMAP_MAGMA)
        return high_freq_mean > self.THRESH_FFT_ENERGY, {"high_freq_energy_mean": float(round(high_freq_mean, 2))}, freq_vis

    # ------------------------------------------------------------
    # MODULE 6: Noise Consistency (Block Variance)
    # ------------------------------------------------------------
    def _analyze_noise_fast(self, cv_gray: np.ndarray) -> Tuple[bool, Dict[str, Any], np.ndarray]:
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
            return False, {"noise_ratio": 1.0, "note": "Blocks too solid/black"}, noise_vis
        min_var = np.percentile(valid_variances, 10)
        max_var = np.percentile(valid_variances, 90)
        ratio = min_var / max_var if max_var > 0 else 1.0
        risk = ratio < self.THRESH_NOISE_RATIO
        return risk, {"noise_consistency_ratio": float(round(ratio, 2))}, noise_vis

    # ------------------------------------------------------------
    # MODULE 7: Fuzzy Logic OCR
    # ------------------------------------------------------------
    def _analyze_ocr_logic(self, cv_gray: np.ndarray, cv_img: np.ndarray) -> Tuple[bool, Dict[str, Any], np.ndarray]:
        h, w = cv_gray.shape
        scale_ratio = 1.0
        if w > 1000:
            scale_ratio = 1000 / w
            cv_gray = cv2.resize(cv_gray, (1000, int(h * scale_ratio)), interpolation=cv2.INTER_LINEAR)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast = clahe.apply(cv_gray)
        thresh = cv2.adaptiveThreshold(contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)

        # OCR with error handling
        try:
            text = pytesseract.image_to_string(thresh, config=self.TESSERACT_CONFIG)
            d = pytesseract.image_to_data(thresh, config=self.TESSERACT_CONFIG, output_type=pytesseract.Output.DICT)
            confs = [int(c) for c in d['conf'] if c != '-1']
            avg_conf = float(np.mean(confs)) if confs else 0.0
        except Exception as e:
            # OCR engine not available
            text = ""
            d = {'text': [], 'left': [], 'top': [], 'width': [], 'height': [], 'conf': []}
            avg_conf = 0.0

        text_upper = text.upper()
        nik_match = re.search(r'\b\d{16}\b', text)
        dob_match = re.search(r'\b(\d{2})-(\d{2})-(\d{4})\b', text)
        sex_match = re.search(r'\b(LAKI-LAKI|PEREMPUAN)\b', text_upper)

        details = {
            "nik_extracted": nik_match.group(0) if nik_match else None,
            "dob_extracted": dob_match.group(0) if dob_match else None,
            "sex_extracted": sex_match.group(0) if sex_match else None,
            "raw_text": text,
            "avg_conf": round(avg_conf, 2),
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
                    details["logic_notes"].append(f"OCR Error/Typo ignored: NIK({nik_date_str}) vs DOB({dob_date_str})")
            elif diff_count == 1:
                details["logic_notes"].append("OCR typo 1 digit (forgiven).")

            if details["sex_extracted"]:
                expected_sex = "PEREMPUAN" if is_female_nik else "LAKI-LAKI"
                if details["sex_extracted"] != expected_sex:
                    risk = True
                    details["logic_fail"] = True
                    details["logic_notes"].append(f"Logic Mismatch: NIK Gender vs Text ({details['sex_extracted']})")

        # Visualization bounding boxes
        vis_img = cv_img.copy()
        if d['text']:
            for i in range(len(d['text'])):
                if int(d['conf'][i]) > 60 and len(d['text'][i].strip()) > 2:
                    x = int(d['left'][i] / scale_ratio)
                    y = int(d['top'][i] / scale_ratio)
                    w_box = int(d['width'][i] / scale_ratio)
                    h_box = int(d['height'][i] / scale_ratio)
                    cv2.rectangle(vis_img, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

        return risk, details, cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------
    # MODULE 8: Risk Engine (Weighted Score)
    # ------------------------------------------------------------
    def _risk_engine(self, quality: Dict, meta: bool, copy: bool, ela: bool,
                     freq: bool, noise: bool, ocr: bool,
                     meta_details: Dict, ocr_details: Dict) -> Tuple[float, str, List[str]]:
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
            flags.append("Texture Inconsistency (Possible Splicing)")
        if freq:
            score += (15 * multiplier)
            flags.append("Frequency Anomaly (AI/Print Pattern)")
        if quality["is_poor_quality"]:
            flags.append("Image Quality Poor (Blur/Exposure)")
            if score < 20:
                score += 15
        final_score = min(score, 100.0)
        if final_score >= 70:
            judgment = "PALSU / MANIPULASI"
        elif final_score >= 35:
            judgment = "SUSPECT (Manual Review)"
        else:
            judgment = "ASLI"
        return final_score, judgment, flags


# ==========================================
# FORENSIC DASHBOARD UI
# ==========================================
class ForensicDashboard:
    """
    Premium enterprise dashboard for the KTP Forensic Detector.
    All UI elements, CSS, and logic reside in this single class.
    """
    def __init__(self) -> None:
        self.analyzer = self._load_analyzer()

    @staticmethod
    @st.cache_resource
    def _load_analyzer() -> KTPForensicAnalyzer:
        return KTPForensicAnalyzer()

    # ------------------------------------------------------------
    # CSS Injection
    # ------------------------------------------------------------
    def _inject_css(self) -> None:
        css = """
        <style>
        /* Global */
        .stApp {
            background: linear-gradient(135deg, #0B1121 0%, #0F172A 100%);
        }
        header[data-testid="stHeader"] {
            background: transparent;
        }
        /* Header */
        .header-container {
            background: linear-gradient(135deg, #1E3A8A 0%, #6D28D9 100%);
            border-radius: 0 0 30px 30px;
            padding: 2rem 1rem;
            text-align: center;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 10px 20px rgba(0,0,0,0.5);
        }
        .header-container h1 {
            font-weight: 800;
            letter-spacing: -1px;
            font-size: 3rem;
            margin-bottom: 0.25rem;
        }
        /* Cards */
        .card {
            background: #1E293B;
            border-radius: 16px;
            padding: 20px;
            margin: 10px 0;
            border: 1px solid #334155;
            box-shadow: 0 8px 16px rgba(0,0,0,0.4);
            color: #F8FAFC;
        }
        .result-status {
            font-size: 2.5rem;
            font-weight: 800;
            text-align: center;
            margin: 10px 0;
        }
        /* Metric overrides */
        div[data-testid="stMetric"] {
            background: #1E293B;
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.4);
            border: 1px solid #334155;
        }
        div[data-testid="stMetric"] label {
            color: #94A3B8;
            font-size: 0.9rem;
            font-weight: 600;
        }
        div[data-testid="stMetricValue"] {
            color: #F8FAFC;
            font-size: 2rem;
            font-weight: 700;
        }
        /* Progress bar */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #F59E0B, #EF4444);
            border-radius: 10px;
        }
        /* Sidebar */
        .stSidebar {
            background: #0F172A;
            border-right: 1px solid #1E293B;
        }
        /* File uploader dropzone */
        section[data-testid="stFileUploaderDropzone"] {
            background: #1E293B;
            border: 2px dashed #475569;
            border-radius: 16px;
            padding: 2rem;
        }
        section[data-testid="stFileUploaderDropzone"]:hover {
            border-color: #3B82F6;
        }
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            background: #1E293B;
            border-radius: 8px 8px 0 0;
            padding: 0.75rem 1.25rem;
            color: #94A3B8;
        }
        .stTabs [aria-selected="true"] {
            background: #334155 !important;
            color: #F8FAFC !important;
        }
        /* Footer */
        .footer {
            margin-top: 3rem;
            padding: 1.5rem;
            background: #1E293B;
            border-radius: 16px 16px 0 0;
            color: #94A3B8;
            text-align: center;
            font-size: 0.85rem;
        }
        /* General info text */
        .info-text {
            color: #CBD5E1;
        }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

    # ------------------------------------------------------------
    # Header
    # ------------------------------------------------------------
    def render_header(self) -> None:
        st.markdown("""
        <div class="header-container">
            <h1>🛡️ KTP FORENSIC AI</h1>
            <p style="font-size:1.2rem; opacity:0.9; margin:0;">Digital Identity Fraud Detection Platform</p>
            <p style="font-size:0.9rem; opacity:0.7; margin-top:0.5rem;">Version 1.0 | Powered by OpenCV & AI</p>
        </div>
        """, unsafe_allow_html=True)

    # ------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------
    def render_sidebar(self, result: Optional[Dict[str, Any]] = None) -> None:
        with st.sidebar:
            st.markdown("<h1 style='text-align: center; color:#F8FAFC;'>🛡️ KTP AI</h1>", unsafe_allow_html=True)
            st.markdown("---")
            with st.expander("📌 About"):
                st.markdown("""
                **KTP Forensic AI** mendeteksi pemalsuan KTP menggunakan teknik Computer Vision klasik.
                Analisis mencakup kualitas gambar, metadata, ELA, FFT, noise, dan validasi logika OCR.
                """)
            with st.expander("📖 Cara Penggunaan"):
                st.markdown("""
                1. Unggah gambar KTP (JPG/JPEG/PNG).
                2. Sistem otomatis menjalankan modul forensik.
                3. Skor risiko & vonis ditampilkan.
                4. Telusuri tab untuk detail mendalam.
                """)
            st.markdown("**📄 Supported Formats:** JPG, JPEG, PNG")
            with st.expander("⚙️ Thresholds Used"):
                thresholds = self.analyzer.thresholds
                for k, v in thresholds.items():
                    st.markdown(f"- {k}: **{v}**")
            st.markdown("**🧠 Engine:** KTPForensicAnalyzer v1.0")
            st.markdown(f"**💻 System:** {platform.system()} {platform.machine()}")
            st.markdown(f"**🐍 Python:** {sys.version.split()[0]}")
            st.markdown(f"**📷 OpenCV:** {cv2.__version__}")
            try:
                tesseract_ver = pytesseract.get_tesseract_version()
                st.markdown(f"**🔤 Tesseract:** {tesseract_ver}")
            except Exception:
                st.markdown("**🔤 Tesseract:** Not found")
            if result and "processing_time_ms" in result:
                st.markdown(f"**⏱️ Last Processing:** {result['processing_time_ms']:.1f} ms")

    # ------------------------------------------------------------
    # File Upload & Info
    # ------------------------------------------------------------
    def render_file_upload(self) -> Any:
        return st.file_uploader(
            "🖼️ Drop KTP image here",
            type=["jpg", "jpeg", "png"],
            key="file_uploader",
            help="Format JPG, JPEG, atau PNG"
        )

    def render_file_info(self, uploaded_file: Any) -> None:
        # Calculate file info
        file_bytes = uploaded_file.getvalue()
        img = Image.open(io.BytesIO(file_bytes))
        name = uploaded_file.name
        size_kb = len(file_bytes) / 1024
        fmt = uploaded_file.type
        w, h = img.size
        st.markdown(f"""
        <div class="card" style="display:flex; justify-content:space-around; text-align:center;">
            <div><strong>📁 Nama</strong><br>{name}</div>
            <div><strong>📏 Resolusi</strong><br>{w} x {h}</div>
            <div><strong>⚖️ Ukuran</strong><br>{size_kb:.1f} KB</div>
            <div><strong>🖼️ Format</strong><br>{fmt}</div>
        </div>
        """, unsafe_allow_html=True)

    def render_empty_state(self) -> None:
        st.markdown("""
        <div class="card" style="text-align:center; padding:3rem;">
            <h2 style="color:#94A3B8;">📂 Belum ada gambar</h2>
            <p class="info-text">Silakan unggah foto KTP untuk memulai analisis forensik.</p>
        </div>
        """, unsafe_allow_html=True)

    # ------------------------------------------------------------
    # Result Card & Risk Score
    # ------------------------------------------------------------
    def render_result_card(self, result: Dict[str, Any]) -> None:
        judgment = result["final_judgment"]
        if "PALSU" in judgment:
            color = "#EF4444"
            bg = "#7F1D1D"
        elif "SUSPECT" in judgment:
            color = "#F59E0B"
            bg = "#78350F"
        else:
            color = "#10B981"
            bg = "#064E3B"
        st.markdown(f"""
        <div class="card" style="text-align:center; background:{bg};">
            <h3 style="margin:0; color:#CBD5E1;">STATUS</h3>
            <div class="result-status" style="color:{color};">{judgment}</div>
            <p style="color:#CBD5E1;">Confidence: {result['confidence_score']}%</p>
        </div>
        """, unsafe_allow_html=True)

        flags = result["triggered_flags"]
        if flags:
            st.markdown("#### 🚨 Anomali / Flags Terdeteksi:")
            for f in flags:
                st.error(f"- {f}")
        else:
            st.success("✅ Tidak ada anomali terdeteksi.")

    def render_risk_score(self, score: float) -> None:
        st.markdown("### 🧮 Risk Score")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(int(score))
        with col2:
            st.metric("Score", f"{score}%")

    # ------------------------------------------------------------
    # Metric Cards
    # ------------------------------------------------------------
    def render_metric_cards(self, metrics: Dict[str, Any]) -> None:
        q = metrics["image_quality"]
        pc = metrics["photocopy_analysis"]
        ela = metrics["compression_ela"]
        freq = metrics["frequency_analysis"]
        noise = metrics["texture_noise"]
        ocr = metrics["ocr_validation"]

        cols = st.columns(4)
        with cols[0]:
            st.metric("🔍 Blur (Laplacian)", f"{q['laplacian_variance']:.1f}", help="Semakin kecil semakin buram")
        with cols[1]:
            st.metric("💡 Brightness", f"{q['brightness']:.1f}")
        with cols[2]:
            st.metric("🎚️ Contrast", f"{q['contrast']:.1f}")
        with cols[3]:
            st.metric("🧩 Entropy", f"{pc['shannon_entropy']:.2f}")

        cols2 = st.columns(4)
        with cols2[0]:
            st.metric("📡 ELA Top 1%", f"{ela['ela_top_1_percentile_mean']:.1f}")
        with cols2[1]:
            st.metric("〰️ Noise Ratio", f"{noise['noise_consistency_ratio']:.2f}")
        with cols2[2]:
            st.metric("📶 FFT Energy", f"{freq['high_freq_energy_mean']:.1f}")
        with cols2[3]:
            st.metric("🧾 OCR Conf", f"{ocr.get('avg_conf', 0):.1f}%")

    # ------------------------------------------------------------
    # Tabs
    # ------------------------------------------------------------
    def render_tabs(self, result: Dict[str, Any]) -> None:
        tabs = st.tabs([
            "📊 Overview", "🖼️ Image Quality", "📸 Metadata",
            "📝 OCR", "〰️ Frequency", "🧩 Noise",
            "🔍 ELA", "📈 Histogram", "🗂️ JSON"
        ])
        with tabs[0]:
            self._tab_overview(result)
        with tabs[1]:
            self._tab_image_quality(result["forensic_metrics"]["image_quality"])
        with tabs[2]:
            self._tab_metadata(result["forensic_metrics"]["metadata"])
        with tabs[3]:
            self._tab_ocr(result["forensic_metrics"]["ocr_validation"])
        with tabs[4]:
            self._tab_frequency(result)
        with tabs[5]:
            self._tab_noise(result)
        with tabs[6]:
            self._tab_ela(result)
        with tabs[7]:
            self._tab_histogram(result["visualizations"]["Histogram_Data"])
        with tabs[8]:
            self._tab_json(result)

    def _tab_overview(self, result: Dict[str, Any]) -> None:
        st.subheader("Ringkasan Forensik")
        cola, colb = st.columns(2)
        with cola:
            st.image(result["visualizations"]["Original"], caption="Original Image", use_column_width=True)
        with colb:
            st.markdown("### Informasi Cepat")
            st.write(f"**Vonis:** {result['final_judgment']}")
            st.write(f"**Skor:** {result['confidence_score']}%")
            st.write(f"**Waktu Proses:** {result['processing_time_ms']} ms")
            flags = result["triggered_flags"]
            if flags:
                st.write("**Flags:**")
                for f in flags:
                    st.markdown(f"- {f}")

        # Forensic metrics table
        st.subheader("Metrik Forensik Lengkap")
        df_data = {
            "Module": [],
            "Key Metric": [],
            "Value": []
        }
        m = result["forensic_metrics"]
        df_data["Module"].append("Quality"); df_data["Key Metric"].append("Laplacian Var"); df_data["Value"].append(m["image_quality"]["laplacian_variance"])
        df_data["Module"].append("Quality"); df_data["Key Metric"].append("Brightness"); df_data["Value"].append(m["image_quality"]["brightness"])
        df_data["Module"].append("Photocopy"); df_data["Key Metric"].append("Saturation Mean"); df_data["Value"].append(m["photocopy_analysis"]["saturation_mean"])
        df_data["Module"].append("Photocopy"); df_data["Key Metric"].append("Entropy"); df_data["Value"].append(m["photocopy_analysis"]["shannon_entropy"])
        df_data["Module"].append("ELA"); df_data["Key Metric"].append("Top 1% Mean"); df_data["Value"].append(m["compression_ela"]["ela_top_1_percentile_mean"])
        df_data["Module"].append("Frequency"); df_data["Key Metric"].append("HF Energy"); df_data["Value"].append(m["frequency_analysis"]["high_freq_energy_mean"])
        df_data["Module"].append("Noise"); df_data["Key Metric"].append("Noise Ratio"); df_data["Value"].append(m["texture_noise"]["noise_consistency_ratio"])
        df_data["Module"].append("OCR"); df_data["Key Metric"].append("Avg Conf"); df_data["Value"].append(m["ocr_validation"].get("avg_conf", 0))
        st.dataframe(df_data, use_container_width=True)

    def _tab_image_quality(self, quality: Dict) -> None:
        st.subheader("Image Quality Metrics")
        st.markdown(f"- Laplacian Variance: **{quality['laplacian_variance']}**")
        st.markdown(f"- Brightness: **{quality['brightness']}**")
        st.markdown(f"- Contrast: **{quality['contrast']}**")
        st.markdown(f"- Poor Quality: **{'Yes' if quality['is_poor_quality'] else 'No'}**")

    def _tab_metadata(self, meta: Dict) -> None:
        st.subheader("EXIF & Metadata Analysis")
        if isinstance(meta, dict) and "status" in meta:
            st.info(meta["status"])
        st.json(meta)

    def _tab_ocr(self, ocr: Dict) -> None:
        st.subheader("OCR & Logic Validation")
        st.markdown(f"**NIK:** {ocr.get('nik_extracted', 'Not found')}")
        st.markdown(f"**DOB:** {ocr.get('dob_extracted', 'Not found')}")
        st.markdown(f"**Gender:** {ocr.get('sex_extracted', 'Not found')}")
        st.markdown(f"**OCR Confidence:** {ocr.get('avg_conf', 0):.1f}%")
        st.markdown("**Logic Notes:**")
        if ocr.get("logic_notes"):
            for note in ocr["logic_notes"]:
                st.warning(note)
        else:
            st.success("Tidak ada masalah logika.")
        st.text_area("Raw OCR Text", value=ocr.get("raw_text", ""), height=150)

    def _tab_frequency(self, result: Dict) -> None:
        st.subheader("Frequency Analysis (FFT)")
        vis = result["visualizations"]
        freq = result["forensic_metrics"]["frequency_analysis"]
        col1, col2 = st.columns(2)
        with col1:
            st.image(vis["FFT_Spectrum"], caption="FFT Magnitude Spectrum", use_column_width=True)
        with col2:
            st.metric("High Freq Energy Mean", f"{freq['high_freq_energy_mean']:.2f}")
            st.markdown("Energi frekuensi tinggi yang mencurigakan dapat mengindikasikan pola cetak atau generasi AI.")

    def _tab_noise(self, result: Dict) -> None:
        st.subheader("Noise Consistency")
        vis = result["visualizations"]
        noise = result["forensic_metrics"]["texture_noise"]
        col1, col2 = st.columns(2)
        with col1:
            st.image(vis["Noise_Map"], caption="Noise Variance Map", use_column_width=True)
        with col2:
            st.metric("Noise Ratio", f"{noise['noise_consistency_ratio']:.3f}")
            st.markdown("Rasio kecil menandakan kemungkinan tempelan atau area hasil manipulasi.")

    def _tab_ela(self, result: Dict) -> None:
        st.subheader("Error Level Analysis")
        vis = result["visualizations"]
        ela = result["forensic_metrics"]["compression_ela"]
        col1, col2 = st.columns(2)
        with col1:
            st.image(vis["ELA_Heatmap"], caption="ELA Heatmap", use_column_width=True)
        with col2:
            st.metric("ELA Top 1% Mean", f"{ela['ela_top_1_percentile_mean']:.2f}")
            st.markdown("Nilai tinggi mengindikasikan adanya perbedaan kompresi (editing).")

    def _tab_histogram(self, hist_data: List) -> None:
        st.subheader("Histogram Grayscale")
        st.bar_chart(hist_data, use_container_width=True)

    def _tab_json(self, result: Dict) -> None:
        st.subheader("Full JSON Result")
        # Remove heavy visualization arrays for cleaner JSON view
        clean_result = {k: v for k, v in result.items() if k != "visualizations"}
        st.json(clean_result)

    # ------------------------------------------------------------
    # Download Section
    # ------------------------------------------------------------
    def render_download_section(self, result: Dict[str, Any]) -> None:
        st.markdown("---")
        st.subheader("📥 Download Hasil")
        col_d1, col_d2, col_d3 = st.columns(3)
        # JSON
        json_str = json.dumps({k: v for k, v in result.items() if k != "visualizations"}, indent=2, default=str)
        col_d1.download_button("📋 Download JSON", data=json_str, file_name="forensic_report.json", mime="application/json")
        # OCR text
        ocr_text = result["forensic_metrics"]["ocr_validation"].get("raw_text", "")
        col_d2.download_button("📄 Download OCR Text", data=ocr_text, file_name="ocr_text.txt")
        # HTML Report
        html_report = self._generate_html_report(result)
        col_d3.download_button("📊 Download Report HTML", data=html_report, file_name="forensic_report.html", mime="text/html")

    def _generate_html_report(self, result: Dict) -> str:
        """Generate simple offline HTML report without extra libraries."""
        flags = "".join(f"<li>{f}</li>" for f in result["triggered_flags"])
        metrics_rows = ""
        m = result["forensic_metrics"]
        for module, key, val in [
            ("Quality", "Laplacian Var", m["image_quality"]["laplacian_variance"]),
            ("Quality", "Brightness", m["image_quality"]["brightness"]),
            ("Photocopy", "Saturation Mean", m["photocopy_analysis"]["saturation_mean"]),
            ("Photocopy", "Entropy", m["photocopy_analysis"]["shannon_entropy"]),
            ("ELA", "Top 1% Mean", m["compression_ela"]["ela_top_1_percentile_mean"]),
            ("Frequency", "HF Energy", m["frequency_analysis"]["high_freq_energy_mean"]),
            ("Noise", "Noise Ratio", m["texture_noise"]["noise_consistency_ratio"]),
            ("OCR", "Avg Conf", m["ocr_validation"].get("avg_conf", 0)),
        ]:
            metrics_rows += f"<tr><td>{module}</td><td>{key}</td><td>{val}</td></tr>"

        html = f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>KTP Forensic Report</title>
<style>
body {{ font-family: 'Segoe UI', sans-serif; background: #f5f5f5; padding: 40px; }}
.card {{ background: white; border-radius: 16px; padding: 24px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 24px; }}
h1 {{ color: #1E3A8A; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background-color: #1E3A8A; color: white; }}
</style>
</head>
<body>
<div class="card">
<h1>🛡️ KTP Forensic Report</h1>
<p><strong>Vonis:</strong> {result['final_judgment']}</p>
<p><strong>Confidence Score:</strong> {result['confidence_score']}%</p>
<p><strong>Processing Time:</strong> {result['processing_time_ms']} ms</p>
</div>
<div class="card">
<h2>🚨 Flags</h2>
<ul>{flags if flags else '<li>No anomalies</li>'}</ul>
</div>
<div class="card">
<h2>📊 Forensic Metrics</h2>
<table><tr><th>Module</th><th>Key Metric</th><th>Value</th></tr>{metrics_rows}</table>
</div>
</body>
</html>"""
        return html

    # ------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------
    def render_footer(self) -> None:
        st.markdown("""
        <div class="footer">
            <p>Powered by <strong>OpenCV</strong> · <strong>Tesseract OCR</strong> · <strong>NumPy</strong> · <strong>Streamlit</strong></p>
            <p style="margin-top:0.5rem;">© 2025 KTP Forensic AI. All rights reserved.</p>
        </div>
        """, unsafe_allow_html=True)

    # ------------------------------------------------------------
    # Main Application Runner
    # ------------------------------------------------------------
    def run(self) -> None:
        st.set_page_config(page_title="KTP Forensic Detector", layout="wide", page_icon="🛡️")
        self._inject_css()
        self.render_header()

        # Sidebar (result might be None initially)
        result = st.session_state.get("forensic_result")
        self.render_sidebar(result)

        # Main content
        uploaded_file = self.render_file_upload()

        # Clear result if no file
        if uploaded_file is None:
            if "forensic_result" in st.session_state:
                del st.session_state.forensic_result
            if "uploaded_file_name" in st.session_state:
                del st.session_state.uploaded_file_name
            self.render_empty_state()
            self.render_footer()
            return

        # Handle new file upload
        if ("uploaded_file_name" not in st.session_state or
                st.session_state.uploaded_file_name != uploaded_file.name):
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.file_bytes = uploaded_file.getvalue()
            with st.spinner("🔍 Menjalankan analisis forensik..."):
                analysis_result = self.analyzer.analyze(st.session_state.file_bytes)
            if analysis_result["status"] == "SUCCESS":
                st.session_state.forensic_result = analysis_result
                st.toast("✅ Analisis selesai!", icon="🛡️")
            else:
                st.error(f"Gagal memproses gambar: {analysis_result.get('message')}")
                st.session_state.forensic_result = None

        # Always show file info if we have it
        if "file_bytes" in st.session_state:
            self.render_file_info(uploaded_file)

        # Render results if available
        if "forensic_result" in st.session_state and st.session_state.forensic_result:
            result = st.session_state.forensic_result
            self.render_result_card(result)
            self.render_risk_score(result["confidence_score"])
            self.render_metric_cards(result["forensic_metrics"])
            self.render_tabs(result)
            self.render_download_section(result)

        self.render_footer()


# ==========================================
# Entry Point
# ==========================================
if __name__ == "__main__":
    dashboard = ForensicDashboard()
    dashboard.run()