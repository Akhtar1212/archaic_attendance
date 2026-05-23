import cv2
import os

def capture_faces(id_pegawai, nama):
    """
    Capture wajah pegawai dari webcam eksternal (index 1).
    Menyimpan 30 sample wajah ke folder dataset/<id_pegawai>/
    """
    cam_index = 1  # kamera eksternal
    cam = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    cam.set(3, 640)
    cam.set(4, 480)

    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    folder_path = os.path.join('dataset', id_pegawai)
    os.makedirs(folder_path, exist_ok=True)

    print(f"[INFO] Mulai capture wajah untuk {nama} ({id_pegawai})...")
    count = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            file_path = os.path.join(folder_path, f"{nama}_{str(count)}.jpg")
            cv2.imwrite(file_path, face_img)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, f"{count}/30", (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow('Capture Face', frame)

        k = cv2.waitKey(100) & 0xff
        if k == 27:  # ESC
            break
        elif count >= 30:
            break

    print(f"[INFO] Capture selesai untuk {nama}. Total {count} gambar disimpan.")
    cam.release()
    cv2.destroyAllWindows()
    return folder_path
