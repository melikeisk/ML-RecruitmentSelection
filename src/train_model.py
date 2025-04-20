import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# 📌 Görselleştirme stilini ayarla
sns.set_theme(style="whitegrid")

# 📂 Veriyi yükle
df = pd.read_csv("data/raw/applicant_data.csv")

# 🎯 Özellikler (X) ve hedef değişken (y)
X = df[["experience_years", "technical_score"]]
y = df["label"]

# 🔀 Veriyi eğitim ve test olarak ayır (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ⚖️ Veriyi ölçekle (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🧠 SVM modeli oluştur ve eğit
model = SVC(kernel="linear", random_state=42)
model.fit(X_train_scaled, y_train)

# 💾 Modeli ve scaler'ı kaydet
os.makedirs("models", exist_ok=True)
with open("models/trained_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# 🔍 Tahmin yap
y_pred = model.predict(X_test_scaled)

# 📊 Metrikleri hesapla
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# 📁 Çıktı klasörünü oluştur
os.makedirs("outputs/models", exist_ok=True)

# 📝 Metrikleri dosyaya yaz
with open("outputs/models/metrics.txt", "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n\n")
    f.write("Classification Report:\n")
    f.write(report)

# 📈 Tahmin vs Gerçek değer scatter plot
plt.figure(figsize=(6, 5))
plt.scatter(X_test["experience_years"], X_test["technical_score"],
            c=y_pred, cmap="bwr", edgecolors="k")
plt.title("Prediction Scatter Plot")
plt.xlabel("Tecrübe (Yıl)")
plt.ylabel("Teknik Puan")
plt.savefig("outputs/models/pred_vs_true.png")
plt.close()

# 📉 Confusion Matrix görseli
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.title("Confusion Matrix")
plt.savefig("outputs/models/confusion_matrix.png")
plt.close()
