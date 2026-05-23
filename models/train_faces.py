# models/train_faces.py
import os, cv2, numpy as np
from datetime import datetime
from .model_state import start_running, bump_progress, finish, fail

# lokasi trainer
MODEL_DIR = os.environ.get("ARCHAIC_MODEL_DIR", "model")
os.makedirs(MODEL_DIR, exist_ok=True)
TRAINER_PATH = os.path.join(MODEL_DIR, "trainer_lbph.yml")

# lokasi dataset
DATASET_DIR = os.environ.get("ARCHAIC_DATASET_DIR", "datasets")

# cascade untuk deteksi (biar aman retrim wajah saat training)
DEFAULT_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

def _load_images():
    people = []       # [(id_pegawai, [img_paths...]), ...]
    total_images = 0
    if not os.path.isdir(DATASET_DIR):
        return [], 0
    for pid in sorted(os.listdir(DATASET_DIR)):
        pdir = os.path.join(DATASET_DIR, pid)
        if not os.path.isdir(pdir): continue
        imgs = [os.path.join(pdir,f) for f in os.listdir(pdir)
                if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))]
        if imgs:
            people.append((pid, imgs))
            total_images += len(imgs)
    return people, total_images

def train_all_faces(cascade_path: str = DEFAULT_CASCADE):
    try:
        start_running()
        people, total = _load_images()
        if not people or total == 0:
            fail("Dataset kosong. Tambahkan data wajah dulu.")
            return 0

        # init
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            fail("Gagal memuat Haar Cascade.")
            return 0

        recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)  # akurat & cepat
        faces, labels = [], []

        processed = 0
        for idx, (pid, img_paths) in enumerate(people, start=1):
            label_int = _label_from_id(pid)  # map ID string -> int
            for p in img_paths:
                img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                # deteksi wajah, crop tengah (fallback: pakai full)
                dets = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
                if len(dets) == 0:
                    roi = _center_crop(img, size=200)
                else:
                    x,y,w,h = sorted(dets, key=lambda t:t[2]*t[3], reverse=True)[0]  # ambil yang paling besar
                    roi = img[y:y+h, x:x+w]
                roi = cv2.resize(roi, (200,200))
                faces.append(roi)
                labels.append(label_int)
                processed += 1
                # progress
                bump_progress(int(processed / max(1,total) * 90))  # 0..90% untuk load

        if not faces:
            fail("Tidak ada wajah valid di dataset.")
            return 0

        recognizer.train(faces, np.array(labels))
        recognizer.write(TRAINER_PATH)
        bump_progress(95)

        finish(trained_people=len(people))
        return len(people)

    except Exception as e:
        fail(f"Training error: {e}")
        return 0

# ---------- util ----------
def _label_from_id(pid: str) -> int:
    # Map ID string ke integer yang stabil
    # contoh: KSR01 -> 10101, BRS02 -> 20202, dst agar beda role gak tabrakan
    prefix_map = {"KSR":1, "BRS":2, "CHF":3, "WTR":4}
    prefix = pid[:3].upper()
    num = 0
    try:
        num = int(pid[3:])
    except:
        # fallback: hash
        num = abs(hash(pid)) % 10000
    base = prefix_map.get(prefix, 9)
    return base*10000 + num

def _center_crop(img, size=200):
    h, w = img.shape[:2]
    s = min(h,w)
    y1 = (h - s)//2
    x1 = (w - s)//2
    crop = img[y1:y1+s, x1:x1+s]
    return cv2.resize(crop, (size,size))
