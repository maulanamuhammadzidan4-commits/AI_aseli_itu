import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load data fitur
df = pd.read_csv('extracted_features.csv')

# Pisahkan Fitur (X) dan Label (y)
X = df.iloc[:, :-1]  # Seluruh kolom kecuali kolom terakhir
y = df.iloc[:, -1]   # Kolom terakhir (label: depan, kanan, kiri)

# 2. Bagi dataset menjadi Data Latih (80%) dan Data Uji (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Latih Model Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluasi Performa Model
y_pred = model.predict(X_test)
print(f"Akurasi Model: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nLaporan Klasifikasi:\n", classification_report(y_test, y_pred))

# 5. Simpan Model AI yang sudah dilatih
with open('face_pose_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model berhasil disimpan sebagai 'face_pose_model.pkl'")