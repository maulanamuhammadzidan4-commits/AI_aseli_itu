import cv2
import mediapipe as mp
import numpy as np
import csv
import os

DATASET_DIR = "dataset"
LABELS_FILE = os.path.join(DATASET_DIR, "labels.csv")
MODEL_FILE = "face_landmarker.task"
OUTPUT_FILE = "features.npz"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_FILE
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)
features = []
labels = []

with FaceLandmarker.create_from_options(options) as landmarker:

    with open(
        LABELS_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)
        for row in reader:
            filename = row["filename"]
            label = row["label"]
            image_path = os.path.join(
                DATASET_DIR,
                filename
            )
            image = cv2.imread(image_path)
            if image is None:
                print(f"Gagal membaca: {filename}")
                continue
            image_rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=image_rgb
            )
            result = landmarker.detect(mp_image)
            if not result.face_landmarks:

                print(
                    f"Wajah tidak terdeteksi: {filename}"
                )

                continue
            face = result.face_landmarks[0]
            landmark_data = []
            for landmark in face:
                landmark_data.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])
            landmark_data = np.array(
                landmark_data,
                dtype=np.float32
            )
            features.append(landmark_data)
            labels.append(label)
            print(
                f"Berhasil: {filename} -> {label}"
            )
features = np.array(
    features,
    dtype=np.float32
)
labels = np.array(labels)
np.savez(
    OUTPUT_FILE,
    features=features,
    labels=labels
)
print()
print("==============================")
print("EKSTRAKSI SELESAI")
print("==============================")
print("Jumlah data  :", len(features))
print("Jumlah fitur :", features.shape[1])
print("Output       :", OUTPUT_FILE)