# models/rules.py
from datetime import datetime, time, timedelta

# Konfigurasi aturan
MULAI_MASUK = time(11, 0, 0)       # 11:00
GRACE_MASUK_MIN = 10               # 10 menit
MULAI_PULANG_WEEKDAY = time(23, 0) # 23:00
MULAI_PULANG_WEEKEND = time(0, 0)  # 00:00 (hanya Sabtu)
EARLY_OUT_ALLOW_MIN = 15           # boleh 15 menit lebih awal
# Minggu termasuk weekday -> weekend hanya Sabtu

def is_weekend_like(dt: datetime) -> bool:
    # Python: Monday=0 ... Sunday=6
    # Weekend (khusus permintaan): hanya Sabtu (5)
    return dt.weekday() == 5

def classify_check_in(now: datetime) -> str:
    batas_tepat = (datetime.combine(now.date(), MULAI_MASUK)
                   + timedelta(minutes=GRACE_MASUK_MIN))
    return "Tepat Waktu" if now <= batas_tepat else "Terlambat"

def allowed_check_out_start(now: datetime) -> datetime:
    if is_weekend_like(now):
        target = datetime.combine(now.date(), MULAI_PULANG_WEEKEND)
        # jam 00:00 adalah awal hari—artinya pulang “malam ini” = 00:00 hari berikut
        if target <= now:
            return target  # sudah lewat
        # kalau belum lewat (siang harinya), ini artinya target malam berikut (rolling)
        return target  # tetap 00:00 di hari yang sama (interpretasi praktis)
    else:
        return datetime.combine(now.date(), MULAI_PULANG_WEEKDAY)

def classify_check_out(now: datetime) -> str:
    start = allowed_check_out_start(now) - timedelta(minutes=EARLY_OUT_ALLOW_MIN)
    # <= start: “Pulang Awal (dalam toleransi)”, > start: “Pulang Normal”
    return "Pulang Awal" if now < start + timedelta(minutes=EARLY_OUT_ALLOW_MIN) else "Pulang Normal"

def can_check_out(now: datetime) -> bool:
    start = allowed_check_out_start(now) - timedelta(minutes=EARLY_OUT_ALLOW_MIN)
    return now >= start

def discipline_points(status_in: str, status_out: str|None) -> int:
    """
    Skor sederhana:
      Tepat Waktu = +2
      Terlambat   = +0
      Pulang Normal = +1
      Pulang Awal   = +0
    """
    pts = 0
    if status_in == "Tepat Waktu":
        pts += 2
    if status_out == "Pulang Normal":
        pts += 1
    return pts
