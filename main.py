import cv2
import mediapipe as mp
import numpy as np

MODEL_FILE = "face_landmarker.task"
TRAINED_MODEL = "face_direction_model.npz"

data = np.load(TRAINED_MODEL)

classes = data["classes"]
centroids = data["centroids"]

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

landmarker = FaceLandmarker.create_from_options(options)

kamera = cv2.VideoCapture(0)

if not kamera.isOpened():
    print("Kamera tidak dapat dibuka.")
    exit()

while True:
    try:
        berhasil, frame = kamera.read()

        if not berhasil:
            print("Gagal membaca kamera.")
            break
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )
        result = landmarker.detect(mp_image)

        arah = "TIDAK TERDETEKSI"

        if result.face_landmarks:

            face = result.face_landmarks[0]
            landmark_data = []

            for landmark in face:
                landmark_data.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            fitur = np.array(
                landmark_data,
                dtype=np.float32
            )
            jarak = []

            for centroid in centroids:
                distance = np.linalg.norm(
                    fitur - centroid
                )
                jarak.append(distance)

            index_terdekat = np.argmin(jarak)

            arah = classes[index_terdekat].upper()


        cv2.putText(
            frame,
            f"Arah: {arah}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

        cv2.imshow(
            "Deteksi Arah Wajah",
            frame
        )
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    except KeyboardInterrupt:
        print("Program dihentikan.")
        break

kamera.release()
cv2.destroyAllWindows()
landmarker.close()