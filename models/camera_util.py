import cv2

def choose_camera(preferred=None, max_index=5):
    """
    Pilih kamera yang tersedia:
    - jika preferred (int) diberikan, coba itu dulu
    - lalu coba urutan umum: 1,0,2,3,4...
    """
    tried = []
    if isinstance(preferred, int):
        tried.append(preferred)
    # urutan umum: eksternal(1) dulu, lalu internal(0), lalu lainnya
    for i in [1, 0] + list(range(2, max_index+1)):
        if i not in tried:
            tried.append(i)
    for idx in tried:
        cam = cv2.VideoCapture(idx)
        if cam.isOpened():
            return cam, idx
        cam.release()
    return None, None
