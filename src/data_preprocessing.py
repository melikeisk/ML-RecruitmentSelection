import os
import pandas as pd
import random
from faker import Faker # Faker kütüphanesi kullanarak sahte veri üretimi için

def generate_applicant_data(save=True):
   # Faker ile sahte aday verisi üretir.
    fake = Faker()
    random.seed(42)

    data = []
    for _ in range(200):
        name = fake.name()
        experience_years = random.randint(0, 10)
        technical_score = random.randint(0, 100)

        # Etiketleme kuralı: tecrübe < 2 ve puan < 60 ise işe alınmaz
        if experience_years < 2 and technical_score < 60:
            label = 1  # işe alınmadı
        else:
            label = 0  # işe alındı

        data.append([name, experience_years, technical_score, label])

    df = pd.DataFrame(data, columns=["name", "experience_years", "technical_score", "label"])

    if save:
        os.makedirs("data/raw", exist_ok=True)
        df.to_csv("data/raw/applicant_data.csv", index=False)
        print("Veri başarıyla kaydedildi: data/raw/applicant_data.csv")

    return df

if __name__ == "__main__":
    generate_applicant_data()
