# models/model_state.py
import json, os, threading, time
from datetime import datetime

STATE_PATH = os.environ.get("ARCHAIC_MODEL_STATE", "model_state.json")
_LOCK = threading.Lock()

DEFAULT_STATE = {
    "status": "idle",             # idle | running | done | error
    "progress": 0,                # 0..100
    "message": "",
    "trained_people": 0,
    "last_trained_at": None
}

def _load():
    if not os.path.exists(STATE_PATH):
        return DEFAULT_STATE.copy()
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = DEFAULT_STATE.copy()
    # lengkapi key hilang
    for k,v in DEFAULT_STATE.items():
        data.setdefault(k, v)
    return data

def _save(data):
    tmp = f"{STATE_PATH}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_PATH)

def get_state():
    with _LOCK:
        return _load()

def set_state(**kwargs):
    with _LOCK:
        data = _load()
        data.update(kwargs)
        _save(data)
        return data

def start_running():
    return set_state(status="running", progress=0, message="Menyiapkan dataset...", trained_people=0)

def finish(trained_people: int):
    return set_state(
        status="done", progress=100, message="Training selesai.",
        trained_people=trained_people,
        last_trained_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

def fail(msg: str):
    return set_state(status="error", message=msg)

def bump_progress(p):
    with _LOCK:
        st = _load()
        st["progress"] = max(st.get("progress",0), min(100, int(p)))
        _save(st)
        return st
