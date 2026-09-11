import cv2
import pickle
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Load Model yang Sudah Dilatih
with open('face_pose_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 2. Inisialisasi FaceLandmarker untuk Video
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Dapatkan stempel waktu (timestamp) dalam milidetik
    frame_timestamp_ms = int(time.time() * 1000)
    detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)
    
    prediction_text = "Mencari Wajah..."
    
    if detection_result.face_landmarks:
        face_landmarks = detection_result.face_landmarks[0]
        landmarks = []
        
        for lm in face_landmarks:
            landmarks.extend([lm.x, lm.y, lm.z])
            
        # Prediksi arah wajah menggunakan model ML
        prediction = model.predict([landmarks])
        prediction_text = f"Arah Wajah: {prediction[0]}"

    cv2.putText(
        frame, prediction_text, (20, 50), 
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )
    
    cv2.imshow('Face Movement Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()