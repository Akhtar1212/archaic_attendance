import os
import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "database.db"

def export_excel(month):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT 
            a.id_presensi,
            a.id_pegawai,
            p.nama,
            p.jabatan,
            a.tanggal,
            a.jam_masuk,
            a.jam_pulang,
            a.status
        FROM presensi a
        LEFT JOIN pegawai p ON a.id_pegawai = p.id_pegawai
        WHERE strftime('%Y-%m', a.tanggal) = ?
        ORDER BY a.tanggal ASC
    """, (month,))
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=[
        "ID Presensi", "ID Pegawai", "Nama", "Jabatan",
        "Tanggal", "Masuk", "Pulang", "Status"
    ])

    export_dir = "exports"
    os.makedirs(export_dir, exist_ok=True)
    filename = f"{export_dir}/rekap_{month}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    df.to_excel(filename, index=False)
    return filename
