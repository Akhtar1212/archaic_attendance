# jalankan: python service_windows.py install
# lalu:     python service_windows.py start
# stop:     python service_windows.py stop
# remove:   python service_windows.py remove

import win32serviceutil, win32service, win32event, servicemanager, subprocess, sys, os, signal

PYTHON = sys.executable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class ArchaicAttendanceService(win32serviceutil.ServiceFramework):
    _svc_name_ = "ArchaicAttendance"
    _svc_display_name_ = "Archaic Attendance (Face Recognition)"
    _svc_description_ = "Archaic Coffee - Absensi berbasis wajah (Flask + Waitress)."

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.stop_event = win32event.CreateEvent(None, 0, 0, None)
        self.proc = None

    def SvcDoRun(self):
        servicemanager.LogInfoMsg("ArchaicAttendanceService starting...")
        env = os.environ.copy()
        cmd = [PYTHON, "-m", "waitress", "--listen=0.0.0.0:5000", "app:app"]
        self.proc = subprocess.Popen(cmd, cwd=BASE_DIR, env=env)
        win32event.WaitForSingleObject(self.stop_event, win32event.INFINITE)

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()
        win32event.SetEvent(self.stop_event)
        self.ReportServiceStatus(win32service.SERVICE_STOPPED)

if __name__ == '__main__':
    win32serviceutil.HandleCommandLine(ArchaicAttendanceService)
