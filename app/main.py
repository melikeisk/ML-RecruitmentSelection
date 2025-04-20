from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

# FastAPI uygulamasını başlat
app = FastAPI(
    title="İşe Alım Tahmin API",
    description="Adayın işe alınıp alınmayacağını tahmin eden bir makine öğrenmesi modeli.",
    version="1.0.0",
    docs_url="/docs",         # Swagger arayüz yolu
    redoc_url=None,           # ReDoc arayüzünü devre dışı bırak
    openapi_url="/openapi.json"
)

# Ana dizin ve model yolları
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "trained_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# Model ve scaler'ı yükle
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Giriş verisi modeli
class Candidate(BaseModel):
    experience_years: float
    technical_score: float

# Tahmin endpoint'i
@app.post("/predict", tags=["Tahmin"])
def predict(candidate: Candidate):
    input_data = np.array([[candidate.experience_years, candidate.technical_score]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    # 1 = RED, 0 = KABUL
    result = "Aday büyük ihtimalle REDDEDİLECEK ❌" if prediction == 1 else "Aday yüksek ihtimalle İŞE ALINACAK ✅"

    return {
        "prediction": int(prediction),
        "message": result,
        "input": candidate.model_dump()
    }

# Ana sayfa endpoint'i
@app.get("/", tags=["Ana Sayfa"])
def read_root():
    return {
        "başlık": "🎯 İşe Alım Tahmin API'sine Hoş Geldiniz!",
        "açıklama": "Bu API, adayların deneyim yılı ve teknik test puanına göre işe alınıp alınmayacağını tahmin eder.",
        "kullanım": "➡️ /docs adresinden interaktif dökümana ulaşabilirsiniz."
    }


# Uygulamayı çalıştırmak için terminalde şu komutu kullanın:
# uvicorn app.main:app --reload
# Bu komut, FastAPI uygulamasını başlatır ve otomatik olarak yeniden yükler.