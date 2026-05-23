import os
import json
import time
import sqlite3
from datetime import datetime

import cv2
import numpy as np

# Import path DB dari models.db
from .db import DB_PATH

class FaceEngine:
    def __init__(self, model_path: str, dataset_dir: str):
        self.model_path = model_path
        self.dataset_dir = dataset_dir
        os.makedirs(self.dataset_dir, exist_ok=True)

        base_dir = os.path.dirname(self.dataset_dir)
        self.proofs_dir = os.path.join(base_dir, "proofs")
        os.makedirs(self.proofs_dir, exist_ok=True)

        # LBPH recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.label_map_path = self.model_path + ".labels.json"
        
        # CACHE: Simpan di RAM agar tidak baca disk terus-menerus
        self.label_map = {} 
        self.name_cache = {}

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # LOAD DATA KE RAM (Startup)
        self._reload_engine_data()

    def _reload_engine_data(self):
        """Memuat Model LBPH dan Label Map ke RAM satu kali saja"""
        if os.path.exists(self.model_path):
            try:
                self.recognizer.read(self.model_path)
                self.label_map = self._load_label_map_from_disk()
                print("[FaceEngine] Model & Label Map loaded into RAM.")
            except Exception as e:
                print("[FaceEngine] Gagal load model:", e)

    # ========== UTIL INTERNAL ==========

    def _decode_jpeg(self, jpeg_bytes):
        arr = np.frombuffer(jpeg_bytes, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def _detect_biggest_face(self, gray, min_size=(100, 100)):
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=min_size
        )
        if len(faces) == 0: return None
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        return faces[0]

    def _blur_score(self, gray_face):
        return cv2.Laplacian(gray_face, cv2.CV_64F).var()

    def _get_employee_name_cached(self, emp_id: str):
        """Ambil nama dari cache RAM, jika tidak ada baru cari ke DB (Efisien)"""
        if emp_id in self.name_cache:
            return self.name_cache[emp_id]
        
        try:
            # Gunakan timeout agar tidak bentrok dengan proses absen di app.py
            conn = sqlite3.connect(DB_PATH, timeout=10)
            cur = conn.cursor()
            cur.execute("SELECT nama FROM employees WHERE id_pegawai=?", (emp_id,))
            row = cur.fetchone()
            conn.close()
            if row:
                self.name_cache[emp_id] = row[0]
                return row[0]
        except Exception as e:
            print("[FaceEngine] DB Error:", e)
        return emp_id

    def _load_label_map_from_disk(self):
        if not os.path.exists(self.label_map_path): return {}
        try:
            with open(self.label_map_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except: return {}

    def _save_label_map(self, label_map):
        try:
            with open(self.label_map_path, "w", encoding="utf-8") as f:
                json.dump(label_map, f, ensure_ascii=False, indent=2)
            self.label_map = label_map # Update RAM
        except Exception as e:
            print("[FaceEngine] Gagal simpan label map:", e)

    # ========== DATASET CAPTURE ==========

    def capture_dataset(self, emp_id: str, nama: str, count: int = 40, stream_getter=None):
        if stream_getter is None: return False, "stream_getter error"
        emp_dir = os.path.join(self.dataset_dir, emp_id)
        os.makedirs(emp_dir, exist_ok=True)

        saved, attempts = 0, 0
        max_attempts = count * 10
        
        # Threshold kualitas
        MIN_BLUR = 80.0 

        while saved < count and attempts < max_attempts:
            attempts += 1
            jpeg = stream_getter()
            if not jpeg: continue

            frame = self._decode_jpeg(jpeg)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_rect = self._detect_biggest_face(gray)
            
            if face_rect is not None:
                x, y, w, h = face_rect
                face_gray = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                
                # Quality Gate: Hanya simpan jika gambar tajam (tidak blur)
                if self._blur_score(face_gray) >= MIN_BLUR:
                    saved += 1
                    path = os.path.join(emp_dir, f"{emp_id}_{saved:03d}.jpg")
                    cv2.imwrite(path, face_gray)
            time.sleep(0.05)

        return (True, f"Sukses: {saved} gambar") if saved > 0 else (False, "Gagal seleksi")

    # ========== TRAINING ==========

    def train(self):
        image_list, label_list, label_map = [], [], {}
        next_label = 1

        for emp_id in sorted(os.listdir(self.dataset_dir)):
            emp_path = os.path.join(self.dataset_dir, emp_id)
            if not os.path.isdir(emp_path): continue

            label_map[str(next_label)] = emp_id
            for fname in os.listdir(emp_path):
                img = cv2.imread(os.path.join(emp_path, fname), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    image_list.append(img)
                    label_list.append(next_label)
            next_label += 1

        if not image_list: return 0

        self.recognizer.train(image_list, np.array(label_list, dtype=np.int32))
        self.recognizer.save(self.model_path)
        self._save_label_map(label_map)
        
        # Penting: Refresh RAM setelah training
        self._reload_engine_data()
        return len(label_map)

    # ========== RECOGNITION (OPTIMIZED) ==========

    def recognize_jpeg(self, jpeg_bytes):
        """Prediksi Wajah (Optimasi RAM & Akurasi Cahaya)"""
        if jpeg_bytes is None or not self.label_map:
            return None, None, None, None

        frame = self._decode_jpeg(jpeg_bytes)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Peningkatan Kontras: Penting untuk akurasi LBPH di dalam ruangan
        gray = cv2.equalizeHist(gray) 

        face_rect = self._detect_biggest_face(gray)
        if face_rect is None: return None, None, None, None

        x, y, w, h = face_rect
        face_gray = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
        face_color = cv2.resize(frame[y:y+h, x:x+w], (200, 200))

        try:
            # Prediksi langsung dari RAM (Bukan dari file disk)
            label_int, confidence = self.recognizer.predict(face_gray)
        except: return None, None, None, None

        # Threshold Keamanan: 80 (Makin kecil makin ketat)
        if confidence > 80.0:
            return None, "Tidak Dikenali", float(confidence), face_color

        emp_id = self.label_map.get(str(label_int))
        if not emp_id: return None, None, float(confidence), face_color

        # Gunakan Cache RAM untuk nama agar tidak membebani database
        nama = self._get_employee_name_cached(emp_id)
        return emp_id, nama, float(confidence), face_color

    # ========== SAVE PROOF ==========
    def save_proof(self, face_img_bgr, emp_id: str, kind: str, dt: datetime):
        if face_img_bgr is None: return None
        ts = dt.strftime("%Y%m%d_%H%M%S")
        fname = f"{emp_id}_{kind}_{ts}.jpg"
        path_abs = os.path.join(self.proofs_dir, fname)
        cv2.imwrite(path_abs, face_img_bgr)
        return os.path.join("proofs", fname).replace("\\", "/")