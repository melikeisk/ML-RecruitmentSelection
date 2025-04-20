# 📦 Proje Klasör Yapısı

```
ML-RecruitmentSelection/
├── app/                        # FastAPI uygulaması
│   └── main.py                 # API endpoint'lerinin yer aldığı dosya
│
├── data/                       # Veri dosyaları 
│   └── applicant_data.csv
│
├── models/                     # Model ve scaler dosyaları
│   ├── trained_model.pkl       # Eğitilmiş SVM modeli
│   └── scaler.pkl              # Verileri ölçeklemek için kullanılan scaler
│
├── outputs/                    # Görselleştirme ve değerlendirme çıktıları
│   ├── eda/
│   └── models/
│       ├── pred_vs_true.png
│       ├── confusion_matrix.png
│       └── metrics.txt
│
├── src/                        # Veri işleme ve modelleme betikleri
│   └── ...
│
├── README.md                   # Projenin genel tanımı ve kullanım talimatları
└── project_structure.md        # Klasör yapısının açıklaması
```
