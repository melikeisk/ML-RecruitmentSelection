import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Görselleştirme stilini ayarla
sns.set_theme(style="whitegrid")


# Veriyi yükle
df = pd.read_csv("data/raw/applicant_data.csv")

# Kayıt klasörü oluştur (yoksa)
os.makedirs("outputs/eda", exist_ok=True)

# 1. Veri bilgisi
print("Veri Şekli:", df.shape)
print("\nİlk 5 Satır:\n", df.head())
print("\nİstatistiksel Özet:\n", df.describe())

# 2. Label dağılımı
plt.figure(figsize=(5, 4))
sns.countplot(x="label", data=df, palette="Set2")
plt.title("Label Dağılımı (0: Alındı, 1: Alınmadı)")
plt.xlabel("Label")
plt.ylabel("Aday Sayısı")
plt.savefig("outputs/eda/label_distribution.png")
plt.close()

# 3. Tecrübe dağılımı
plt.figure(figsize=(5, 4))
sns.histplot(df["experience_years"], bins=10, kde=True)
plt.title("Tecrübe Dağılımı")
plt.xlabel("Tecrübe (Yıl)")
plt.savefig("outputs/eda/experience_distribution.png")
plt.close()

# 4. Teknik puan dağılımı
plt.figure(figsize=(5, 4))
sns.histplot(df["technical_score"], bins=10, kde=True, color="orange")
plt.title("Teknik Puan Dağılımı")
plt.xlabel("Teknik Puan")
plt.savefig("outputs/eda/technical_score_distribution.png")
plt.close()

# 5. Scatter plot (label'a göre)
plt.figure(figsize=(6, 5))
sns.scatterplot(
    x="experience_years",
    y="technical_score",
    hue="label",
    palette={0: "green", 1: "red"},
    data=df
)
plt.title("Tecrübe vs Teknik Puan (Label'a Göre)")
plt.xlabel("Tecrübe (Yıl)")
plt.ylabel("Teknik Puan")
plt.legend(title="Label (0: Alındı, 1: Alınmadı)")
plt.savefig("outputs/eda/scatter_experience_vs_score.png")
plt.close()
