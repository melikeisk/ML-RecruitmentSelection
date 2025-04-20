import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data(path="data/raw/applicant_data.csv"):
    """
    Veriyi yükler, eğitim/test olarak böler ve ölçekler.
    Girdi değişkenleri: experience_years, technical_score
    Hedef değişken: label
    """
    # Veriyi oku
    df = pd.read_csv(path)

    # Girdi ve hedef değişkenleri ayır
    X = df[["experience_years", "technical_score"]]
    y = df["label"]

    # Eğitim ve test verisi olarak ayır 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Veriyi ölçekle (StandardScaler: ortalama 0, std sapma 1 olacak şekilde)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
