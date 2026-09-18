import numpy as np

INPUT_FILE = "features.npz"
OUTPUT_FILE = "face_direction_model.npz"

data = np.load(INPUT_FILE)

features = data["features"]
labels = data["labels"]

print("Jumlah data :", len(features))
print("Jumlah fitur:", features.shape[1])

classes = np.unique(labels)

centroids = {}

for label in classes:
    data_class = features[labels == label]

    centroids[label] = np.mean(data_class, axis=0)

    print(f"{label:6} : {len(data_class)} data")

np.savez(
    OUTPUT_FILE,
    classes=classes,
    centroids=np.array(
        [centroids[label] for label in classes]
    )
)

print()
print("==============================")
print("TRAINING SELESAI")
print("==============================")
print("Model disimpan:", OUTPUT_FILE)