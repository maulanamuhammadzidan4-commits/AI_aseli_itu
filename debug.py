import os
import cv2
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_PATH = os.path.join(BASE_DIR, 'dataset/labels.csv')

# Load Model
model_path = os.path.join(BASE_DIR, 'face_landmarker.task')
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

df_labels = pd.read_csv(LABELS_PATH)
print("--- START DEBUGGING ---")
print("Kolom terdeteksi di labels.csv:", list(df_labels.columns))

valid_count = 0

for index, row in df_labels.iterrows():
    # Ambil nama file & label
    file_name = row['filename'] # Pastikan nama kolom sesuai!
    label = row['label']
    
    img_path = os.path.join(BASE_DIR, 'dataset', file_name)
    image = cv2.imread(img_path)
    
    if image is None:
        print(f"[GAGAL BACA] Gambar tidak ditemukan/rusak: {img_path}")
        continue
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    detection_result = detector.detect(mp_image)
    
    if not detection_result.face_landmarks:
        print(f"[TIDAK ADA WAJAH] MediaPipe gagal deteksi wajah di: {file_name}")
        continue
        
    print(f"[BERHASIL] Wajah terdeteksi di: {file_name} -> Label: {label}")
    valid_count += 1

print(f"\nTotal data berhasil diekstrak: {valid_count} dari {len(df_labels)} gambar.")