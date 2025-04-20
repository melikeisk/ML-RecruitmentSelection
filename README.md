# 🧠 ML-RecruitmentSelection

Bu proje, makine öğrenmesi kullanarak bir adayın işe alınıp alınmayacağını tahmin etmeyi amaçlayan bir sınıflandırma modeline dayalıdır. FastAPI kullanılarak bir REST API geliştirilmiştir.

## 📌 Proje Amacı

Adayların **deneyim yılı** ve **teknik test puanı** verilerine göre işe alım kararı (kabul/red) verilir. Bu karar, önceden eğitilmiş bir **Support Vector Machine (SVM)** modeli tarafından yapılır.

## 🚀 API Nasıl Çalışır?

API'ye bir POST isteği ile aday bilgileri gönderilir ve model bu adaya işe alım kararı verir:

### 🔍 Tahmin Endpoint’i
```
POST /predict
```

### 📥 Örnek İstek
```json
{
  "experience_years": 3,
  "technical_score": 85
}
```

### 📤 Örnek Yanıt
```json
{
  "prediction": 0,
  "message": "Aday yüksek ihtimalle İŞE ALINACAK ✅",
  "input": {
    "experience_years": 3,
    "technical_score": 85
  }
}
```

## 🛠 Kurulum ve Çalıştırma

1. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Uygulamayı başlatın:
```bash
uvicorn app.main:app --reload 
```

3. Swagger UI üzerinden API’yi test edin:  
👉 [http://127.0.0.1:8000 ](http://127.0.0.1:8000/docs)

## 🗂 Klasör Yapısı

Projenin klasör yapısı için `project_structure.md` dosyasına göz atabilirsiniz.
