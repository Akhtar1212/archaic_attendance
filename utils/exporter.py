# utils/exporter.py
import os
from datetime import datetime
from io import BytesIO

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# -------- Excel --------
def export_excel(rows, month: str):
    """
    rows: list of [id, id_pegawai, nama, jabatan, tgl, jam_in, jam_out, status, proof_in, proof_out]
    return: (filepath, filename)
    """
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("Package 'pandas' belum terpasang. Install: pip install pandas openpyxl")

    df = pd.DataFrame(rows, columns=[
        "ID","ID Pegawai","Nama","Jabatan","Tanggal","Masuk","Pulang","Status","Bukti Masuk","Bukti Pulang"
    ])

    ensure_dir("exports")
    filename = f"Rekap_{month.replace('-', '')}.xlsx"
    filepath = os.path.join("exports", filename)

    # pakai openpyxl jika tersedia
    try:
        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Rekap")
            ws = writer.sheets["Rekap"]
            for col in ws.columns:
                max_len = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max_len+2, 40)
    except Exception:
        # fallback ke xlsxwriter
        with pd.ExcelWriter(filepath, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Rekap")
            workbook  = writer.book
            worksheet = writer.sheets["Rekap"]
            worksheet.set_column(0, 9, 22)

    return filepath, filename

# -------- PDF --------
def export_pdf(rows, month: str):
    """
    return: (filepath, filename)
    """
    try:
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import getSampleStyleSheet
    except ImportError:
        raise RuntimeError("Package 'reportlab' belum terpasang. Install: pip install reportlab")

    ensure_dir("exports")
    filename = f"Rekap_{month.replace('-', '')}.pdf"
    filepath = os.path.join("exports", filename)

    doc = SimpleDocTemplate(filepath, pagesize=landscape(A4), rightMargin=18, leftMargin=18, topMargin=18, bottomMargin=18)
    styles = getSampleStyleSheet()

    title = Paragraph(f"<b>Rekap Presensi Bulanan — {month}</b>", styles["Title"])
    sub   = Paragraph("Archaic Coffee — Face Recognition", styles["Normal"])

    data = [["ID","ID Pegawai","Nama","Jabatan","Tanggal","Masuk","Pulang","Status","Bukti Masuk","Bukti Pulang"]]
    for r in rows:
        data.append(list(r))

    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#333")),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.HexColor("#d4af37")),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("ALIGN",      (0,0), (-1,0), "CENTER"),
        ("FONTSIZE",   (0,0), (-1,0), 10),

        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#444")),
        ("BACKGROUND", (0,1), (-1,-1), colors.HexColor("#111")),
        ("TEXTCOLOR",  (0,1), (-1,-1), colors.white),
        ("FONTSIZE",   (0,1), (-1,-1), 9),
    ]))

    story = [title, sub, Spacer(1, 8), table]
    doc.build(story)
    return filepath, filename
