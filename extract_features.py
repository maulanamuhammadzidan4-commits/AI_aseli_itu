import os
import cv2
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
LABELS_PATH = os.path.join(BASE_DIR, 'dataset/labels.csv')

# 1. Inisialisasi FaceLandmarker
model_path = os.path.join(BASE_DIR, 'face_landmarker.task')
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# 2. Ambil daftar semua file asli yang ada di folder dataset
actual_files = {f.lower(): f for f in os.listdir(DATASET_DIR)}

df_labels = pd.read_csv(LABELS_PATH)
features_list = []

print("--- MEMULAI PROSES EKSTRAKSI FITUR ---")

for index, row in df_labels.iterrows():
    # Hapus spasi tak terlihat di awal/akhir string
    raw_filename = str(row['filename']).strip()
    label = str(row['label']).strip()
    
    # Cari nama file asli tanpa memedulikan huruf besar/kecil
    matched_filename = actual_files.get(raw_filename.lower())
    
    if not matched_filename:
        print(f"[ERROR] File '{raw_filename}' TIDAK DITEMUKAN di folder dataset!")
        continue
        
    img_path = os.path.join(DATASET_DIR, matched_filename)
    image = cv2.imread(img_path)
    
    if image is None:
        print(f"[ERROR] Gagal membaca gambar (File corrupt/format salah): {matched_filename}")
        continue
        
    # Konversi BGR ke RGB untuk MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    detection_result = detector.detect(mp_image)
    
    if detection_result.face_landmarks:
        face_landmarks = detection_result.face_landmarks[0]
        landmarks = []
        
        for lm in face_landmarks:
            landmarks.extend([lm.x, lm.y, lm.z])
            
        landmarks.append(label)
        features_list.append(landmarks)
        print(f"[BERHASIL] Ekstrak landmark: {matched_filename} -> {label}")
    else:
        print(f"[WARNING] Wajah tidak terdeteksi oleh MediaPipe di file: {matched_filename}")

# 3. Simpan ke CSV
output_path = os.path.join(BASE_DIR, 'extracted_features.csv')
df_features = pd.DataFrame(features_list)
df_features.to_csv(output_path, index=False)

print(f"\nSelesai! Berhasil mengekstrak {len(features_list)} gambar ke 'extracted_features.csv'.")