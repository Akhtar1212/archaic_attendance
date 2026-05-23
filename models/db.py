import sqlite3, os, shutil
from datetime import date, datetime
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database.db")
DB_PATH = os.path.abspath(DB_PATH)
DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets")

SCHEMA = """
CREATE TABLE IF NOT EXISTS employees(
  id_pegawai TEXT PRIMARY KEY,
  nama TEXT NOT NULL,
  jabatan TEXT NOT NULL,
  status TEXT DEFAULT 'Aktif'
);
CREATE TABLE IF NOT EXISTS employees_archive(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  id_pegawai TEXT NOT NULL,
  nama TEXT NOT NULL,
  jabatan TEXT NOT NULL,
  resigned_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS attendance(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  id_pegawai TEXT NOT NULL,
  tanggal TEXT NOT NULL, -- YYYY-MM-DD
  check_in TEXT,   -- HH:MM:SS
  check_out TEXT,  -- HH:MM:SS
  status TEXT,     -- Tepat Waktu / Terlambat
  bukti_in TEXT,
  bukti_out TEXT
);
CREATE TABLE IF NOT EXISTS settings(
  key TEXT PRIMARY KEY,
  value TEXT
);
CREATE TABLE IF NOT EXISTS leave_requests(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  id_pegawai TEXT NOT NULL,
  tanggal TEXT NOT NULL,   -- YYYY-MM-DD
  jenis TEXT NOT NULL,     -- Izin/Cuti/Terlambat
  alasan TEXT,
  status TEXT DEFAULT 'Menunggu', -- Menunggu/Disetujui/Ditolak
  alasan_admin TEXT,
  created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS emp_auth(
  id_pegawai TEXT PRIMARY KEY,
  username TEXT UNIQUE,
  password_hash TEXT
);
"""

def get_conn():
    # check_same_thread=False: Biar kamera & web bisa akses bareng
    # timeout=15: Menghindari error "Database is locked" saat akses bersamaan
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=15)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    with conn:
        conn.executescript(SCHEMA)

def set_setting(key, value):
    conn = get_conn()
    with conn:
        conn.execute("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key,value))

def get_setting(key, default=None):
    conn = get_conn()
    cur = conn.execute("SELECT value FROM settings WHERE key=?", (key,))
    row = cur.fetchone()
    return row["value"] if row else default

# ===== Employees =====
PREFIX = {"Kasir":"KSR","Barista":"BRS","Chef":"CHF","Waitress":"WTR"}

def generate_emp_id(jabatan: str):
    prefix = PREFIX.get(jabatan, "EMP")
    conn = get_conn()
    cur = conn.execute(
        "SELECT id_pegawai FROM employees WHERE id_pegawai LIKE ? ORDER BY id_pegawai",
        (f"{prefix}%",)
    )
    existing = [r["id_pegawai"] for r in cur.fetchall()]
    used = set()
    for e in existing:
        try: used.add(int(e[-2:]))
        except: pass
    for i in range(1, 100):
        if i not in used: return f"{prefix}{i:02d}"
    return f"{prefix}{len(existing)+1:02d}"

def add_employee(idp, nama, jabatan):
    conn = get_conn()
    with conn:
        conn.execute("INSERT OR IGNORE INTO employees(id_pegawai,nama,jabatan,status) VALUES(?,?,?,'Aktif')",
                     (idp, nama, jabatan))

def list_employees():
    conn = get_conn()
    rows = conn.execute("SELECT id_pegawai,nama,jabatan,status FROM employees ORDER BY id_pegawai").fetchall()
    return [tuple(r) for r in rows]

def mark_resign(idp):
    conn = get_conn()
    with conn:
        emp = conn.execute("SELECT id_pegawai,nama,jabatan FROM employees WHERE id_pegawai=?", (idp,)).fetchone()
        if not emp:
            raise ValueError("ID pegawai tidak ditemukan")
        
        conn.execute(
            "INSERT INTO employees_archive(id_pegawai,nama,jabatan,resigned_at) VALUES(?,?,?,datetime('now'))",
            (emp["id_pegawai"], emp["nama"], emp["jabatan"])
        )
        conn.execute("DELETE FROM employees WHERE id_pegawai=?", (idp,))
        conn.execute("DELETE FROM emp_auth WHERE id_pegawai=?", (idp,))

    nama_folder = idp 
    folder_path = os.path.join(DATASET_DIR, nama_folder)
    
    if os.path.exists(folder_path):
        try:
            shutil.rmtree(folder_path)
        except Exception as e:
            print(f"[ERROR] Gagal hapus folder: {e}")

# ===== Credentials =====
def set_credentials(idp, username, password):
    ph = generate_password_hash(password)
    conn = get_conn()
    with conn:
        emp = conn.execute("SELECT 1 FROM employees WHERE id_pegawai=?", (idp,)).fetchone()
        if not emp:
            raise ValueError("ID pegawai tidak ditemukan")
        old = conn.execute("SELECT 1 FROM emp_auth WHERE id_pegawai=?", (idp,)).fetchone()
        if old:
            conn.execute("UPDATE emp_auth SET username=?, password_hash=? WHERE id_pegawai=?", (username, ph, idp))
        else:
            conn.execute("INSERT INTO emp_auth(id_pegawai,username,password_hash) VALUES(?,?,?)", (idp, username, ph))

def get_pegawai_by_credentials(username, password):
    conn = get_conn()
    row = conn.execute("""
        SELECT e.id_pegawai, e.nama, e.jabatan, e.status, a.password_hash
        FROM emp_auth a
        JOIN employees e ON e.id_pegawai = a.id_pegawai
        WHERE a.username = ?
    """, (username,)).fetchone()
    if not row: return None
    if not check_password_hash(row["password_hash"], password): return None
    return (row["id_pegawai"], row["nama"], row["jabatan"], row["status"])

# ===== Attendance (REVISED SECTION) =====

def today_row_for(idp, tgl=None):
    """Mencari baris absensi berdasarkan ID dan Tanggal (default: hari ini)"""
    if tgl is None: 
        tgl = date.today().isoformat()
    conn = get_conn()
    row = conn.execute("SELECT * FROM attendance WHERE id_pegawai=? AND tanggal=?", (idp, tgl)).fetchone()
    return dict(row) if row else None

def upsert_checkin(idp, jam_in, status, bukti_in, tgl=None):
    """Update atau Insert data masuk berdasarkan Tanggal Operasional"""
    if tgl is None: 
        tgl = date.today().isoformat()
    conn = get_conn()
    with conn:
        row = conn.execute("SELECT id FROM attendance WHERE id_pegawai=? AND tanggal=?", (idp, tgl)).fetchone()
        if row:
            conn.execute("UPDATE attendance SET check_in=?, status=?, bukti_in=? WHERE id=?",
                         (jam_in, status, bukti_in, row["id"]))
        else:
            conn.execute("INSERT INTO attendance(id_pegawai,tanggal,check_in,status,bukti_in) VALUES(?,?,?,?,?)",
                         (idp, tgl, jam_in, status, bukti_in))

def upsert_checkout(idp, jam_out, bukti_out, tgl=None):
    """Update atau Insert data pulang berdasarkan Tanggal Operasional"""
    if tgl is None: 
        tgl = date.today().isoformat()
    conn = get_conn()
    with conn:
        row = conn.execute("SELECT id FROM attendance WHERE id_pegawai=? AND tanggal=?", (idp, tgl)).fetchone()
        if row:
            conn.execute("UPDATE attendance SET check_out=?, bukti_out=? WHERE id=?", (jam_out, bukti_out, row["id"]))
        else:
            conn.execute("INSERT INTO attendance(id_pegawai,tanggal,check_out,bukti_out) VALUES(?,?,?,?)",
                         (idp, tgl, jam_out, bukti_out))

# ===== Recap & Lists =====

def month_recap_with_proof(month):
    conn = get_conn()
    sql = """
    SELECT a.id, a.id_pegawai, e.nama, e.jabatan, a.tanggal, a.check_in, a.check_out, a.status, a.bukti_in, a.bukti_out
    FROM attendance a
    LEFT JOIN employees e ON e.id_pegawai = a.id_pegawai
    WHERE a.tanggal LIKE ?
    ORDER BY a.tanggal, a.id_pegawai
    """
    rows = conn.execute(sql, (f"{month}%",)).fetchall()
    return [tuple(r) for r in rows]

def rekap_by_employee(month, idp):
    conn = get_conn()
    sql = """
    SELECT a.id, a.id_pegawai, e.nama, e.jabatan, a.tanggal, a.check_in, a.check_out, a.status, a.bukti_in, a.bukti_out
    FROM attendance a
    LEFT JOIN employees e ON e.id_pegawai = a.id_pegawai
    WHERE a.tanggal LIKE ? AND a.id_pegawai=?
    ORDER BY a.tanggal
    """
    rows = conn.execute(sql, (f"{month}%", idp)).fetchall()
    return [tuple(r) for r in rows]

def list_attendance_by_employee(month, idp):
    conn = get_conn()
    sql = """
    SELECT tanggal, check_in, check_out, status
    FROM attendance
    WHERE tanggal LIKE ? AND id_pegawai=?
    ORDER BY tanggal
    """
    rows = conn.execute(sql, (f"{month}%", idp)).fetchall()
    return [tuple(r) for r in rows]

def list_today_dashboard():
    conn = get_conn()
    sql = """
    SELECT e.id_pegawai, e.nama, e.jabatan,
           (SELECT check_in FROM attendance a WHERE a.id_pegawai=e.id_pegawai AND a.tanggal=date('now')) AS masuk,
           (SELECT check_out FROM attendance a WHERE a.id_pegawai=e.id_pegawai AND a.tanggal=date('now')) AS pulang,
           (SELECT status FROM attendance a WHERE a.id_pegawai=e.id_pegawai AND a.tanggal=date('now')) AS status
    FROM employees e
    ORDER BY e.id_pegawai
    """
    rows = conn.execute(sql).fetchall()
    return [tuple(r) for r in rows]

# ===== Leave / Izin =====
def submit_leave(idp, tanggal, jenis, alasan):
    conn = get_conn()
    with conn:
        conn.execute("INSERT INTO leave_requests(id_pegawai,tanggal,jenis,alasan,status) VALUES(?,?,?,?, 'Menunggu')",
                     (idp, tanggal, jenis, alasan))

def list_leave_by_month(month):
    conn = get_conn()
    sql = """
    SELECT l.id, l.id_pegawai, e.nama, e.jabatan, l.tanggal, l.jenis, l.alasan, l.status, l.alasan_admin, l.created_at
    FROM leave_requests l
    LEFT JOIN employees e ON e.id_pegawai = l.id_pegawai
    WHERE l.tanggal LIKE ?
    ORDER BY l.created_at DESC
    """
    rows = conn.execute(sql, (f"{month}%",)).fetchall()
    return [tuple(r) for r in rows]

def approve_leave(leave_id):
    conn = get_conn()
    with conn:
        conn.execute("UPDATE leave_requests SET status='Disetujui', alasan_admin=NULL WHERE id=?", (leave_id,))

def reject_leave(leave_id, alasan_admin=""):
    conn = get_conn()
    with conn:
        conn.execute("UPDATE leave_requests SET status='Ditolak', alasan_admin=? WHERE id=?", (alasan_admin, leave_id))

# ===== Leaderboard =====
def leaderboard_month(month):
    conn = get_conn()
    sql = """
    SELECT e.id_pegawai, e.nama, e.jabatan,
           SUM(CASE WHEN a.status='Tepat Waktu' THEN 1 ELSE 0 END) AS tepat,
           SUM(CASE WHEN a.status='Terlambat' THEN 1 ELSE 0 END) AS telat,
           COUNT(a.id) AS hadir
    FROM employees e
    LEFT JOIN attendance a ON a.id_pegawai=e.id_pegawai AND a.tanggal LIKE ?
    GROUP BY e.id_pegawai, e.nama, e.jabatan
    ORDER BY (tepat*2 - telat) DESC, tepat DESC, hadir DESC, e.id_pegawai ASC
    """
    rows = conn.execute(sql, (f"{month}%",)).fetchall()
    out = []
    for r in rows:
        score = (r["tepat"] or 0)*2 - (r["telat"] or 0)
        out.append((r["id_pegawai"], r["nama"], r["jabatan"], r["hadir"] or 0, r["tepat"] or 0, r["telat"] or 0, score))
    return out