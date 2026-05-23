# models/capture_wizard.py
import os, cv2, time, threading
from datetime import datetime
import numpy as np

DATASET_DIR = os.environ.get("ARCHAIC_DATASET_DIR", "datasets")
os.makedirs(DATASET_DIR, exist_ok=True)

DEFAULT_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# State global sederhana untuk 1 job pada satu waktu
_CAP_STATE = {
    "running": False,
    "id_pegawai": None,
    "frames_target": 0,
    "frames_captured": 0,
    "message": "",
    "last_file": None,
    "started_at": None,
    "stopped": False
}
_LOCK = threading.Lock()
_THREAD = None

def status():
    with _LOCK:
        return dict(_CAP_STATE)

def stop():
    with _LOCK:
        _CAP_STATE["stopped"] = True
    return True

def _reset_state():
    with _LOCK:
        _CAP_STATE.update({
            "running": False,
            "id_pegawai": None,
            "frames_target": 0,
            "frames_captured": 0,
            "message": "",
            "last_file": None,
            "started_at": None,
            "stopped": False
        })

def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def _save_frame(pid: str, frame):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    dstdir = os.path.join(DATASET_DIR, pid)
    _ensure_dir(dstdir)
    path = os.path.join(dstdir, f"{pid}_{ts}.jpg")
    cv2.imwrite(path, frame)
    return path

def _brightness_ok(gray, lo=60, hi=190):
    """Cek rata2 brightness (0..255)."""
    mean = float(np.mean(gray))
    if mean < lo:
        return False, f"Pencahayaan terlalu gelap (avg={mean:.0f}). Tambah lampu/lebih dekat."
    if mean > hi:
        return False, f"Terlalu terang (avg={mean:.0f}). Jauhkan dari lampu/kurangi exposure."
    return True, ""

def _open_camera(index: int):
    """
    Coba buka kamera berurutan:
    1) MSMF (paling umum di Win10/11)
    2) DSHOW (fallback stabil)
    3) Any/default
    Return (cap, backend_name, error_message)
    """
    trials = [
        (cv2.CAP_MSMF, "MSMF"),
        (cv2.CAP_DSHOW, "DSHOW"),
        (0, "ANY")
    ]
    last_err = None
    for backend, name in trials:
        cap = cv2.VideoCapture(int(index), backend)
        if not cap.isOpened():
            cap.release()
            last_err = f"Gagal open backend {name}"
            continue
        # test read
        ok, _ = cap.read()
        if ok:
            return cap, name, None
        last_err = f"Backend {name} terbuka tapi tidak bisa read()"
        cap.release()
    return None, None, last_err or "Tidak ada backend yang berhasil"

def _capture_job(pid: str, frames_target: int, camera_index: int, cascade_path: str):
    _reset_state()
    with _LOCK:
        _CAP_STATE["running"] = True
        _CAP_STATE["id_pegawai"] = pid
        _CAP_STATE["frames_target"] = int(frames_target)
        _CAP_STATE["frames_captured"] = 0
        _CAP_STATE["message"] = "Menyiapkan kamera..."
        _CAP_STATE["started_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    face_cascade = cv2.CascadeClassifier(cascade_path or DEFAULT_CASCADE)
    if face_cascade.empty():
        with _LOCK:
            _CAP_STATE["message"] = "Gagal memuat Haar Cascade."
            _CAP_STATE["running"] = False
        return

    cap, backend_name, err = _open_camera(camera_index)
    if cap is None:
        with _LOCK:
            _CAP_STATE["message"] = (
                f"Kamera index {camera_index} gagal dibuka. {err}. "
                "Cek: Settings > Scan Kamera, pastikan tidak dipakai app lain, "
                "periksa Privacy Camera Windows, dan driver terpasang."
            )
            _CAP_STATE["running"] = False
        return

    # set resolusi sedang (opsional)
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    except Exception:
        pass

    interval = 0.15  # 150ms antar foto
    min_size = (80,80)

    try:
        with _LOCK:
            _CAP_STATE["message"] = f"Capture berjalan (backend {backend_name}). Hadapkan wajah ke kamera..."

        while True:
            with _LOCK:
                if _CAP_STATE["stopped"]:
                    _CAP_STATE["message"] = "Capture dihentikan."
                    _CAP_STATE["running"] = False
                    break
                done = (_CAP_STATE["frames_captured"] >= _CAP_STATE["frames_target"])

            if done:
                with _LOCK:
                    _CAP_STATE["message"] = "Selesai."
                    _CAP_STATE["running"] = False
                break

            ok, frame = cap.read()
            if not ok:
                with _LOCK:
                    _CAP_STATE["message"] = "Gagal membaca frame dari kamera. Pastikan kamera tidak dipakai aplikasi lain."
                time.sleep(0.2)
                continue

            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dets = face_cascade.detectMultiScale(gray_full, 1.1, 5, minSize=min_size)

            if len(dets) == 0:
                with _LOCK:
                    _CAP_STATE["message"] = "Wajah belum terdeteksi. Arahkan wajah & buka mata."
                time.sleep(0.05)
                continue

            # Ambil wajah terbesar
            x,y,w,h = sorted(dets, key=lambda t:t[2]*t[3], reverse=True)[0]
            roi = frame[y:y+h, x:x+w]
            gray = gray_full[y:y+h, x:x+w]

            ok_b, msg = _brightness_ok(gray, lo=60, hi=190)
            if not ok_b:
                with _LOCK:
                    _CAP_STATE["message"] = f"{msg}"
                time.sleep(0.05)
                continue

            # Normalisasi ukuran
            roi = cv2.resize(roi, (300,300))

            path = _save_frame(pid, roi)
            with _LOCK:
                _CAP_STATE["frames_captured"] += 1
                _CAP_STATE["last_file"] = path.replace("\\","/")
                _CAP_STATE["message"] = f"Captured: {_CAP_STATE['frames_captured']}/{_CAP_STATE['frames_target']}"

            time.sleep(interval)

    finally:
        cap.release()

def start_capture(pid: str, frames_target: int = 40, camera_index: int = 0, cascade_path: str = DEFAULT_CASCADE):
    global _THREAD
    with _LOCK:
        if _CAP_STATE["running"]:
            return False, "Masih ada proses capture yang berjalan."
        _CAP_STATE["message"] = "Menjadwalkan capture..."

    _THREAD = threading.Thread(
        target=_capture_job,
        args=(pid, int(frames_target), int(camera_index), cascade_path),
        daemon=True
    )
    _THREAD.start()
    return True, "Capture dimulai."
