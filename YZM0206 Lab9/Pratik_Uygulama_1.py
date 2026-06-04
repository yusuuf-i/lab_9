import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("=== PRATİK UYGULAMA 1 BAŞLADI ===\n")

# ==========================================
# 1. ORİJİNAL VERİ SETİ İLE ANALİZ (Az Kayıtlı)
# ==========================================
print("--- Bölüm 1: Orijinal Veri Seti (10 Kayıt) Analizi ---")
veri_sozlugu_kucuk = {
    'Musteri_ID': list(range(1, 11)),
    'Yas': [34, 42, np.nan, 23, 55, 40, 28, 48, np.nan, 62],
    'Kredi_Skoru': [650, 580, 710, 620, np.nan, 690, 510, 730, 640, 600],
    'Sehir': ['İstanbul', 'Ankara', 'İstanbul', 'İzmir', 'Ankara', 'İstanbul', 'İzmir', 'Ankara', 'İstanbul', 'İzmir'],
    'Cinsiyet': ['Kadın', 'Erkek', 'Erkek', 'Kadın', np.nan, 'Erkek', 'Kadın', 'Erkek', 'Kadın', 'Erkek'],
    'Terk_Etti_Mi': [0, 1, 0, 0, 1, 0, 1, 1, 0, 1]
}
df_kucuk = pd.DataFrame(veri_sozlugu_kucuk).set_index('Musteri_ID')

def analiz_yap(df, dataset_name):
    X = df.drop(columns=['Terk_Etti_Mi'])
    y = df['Terk_Etti_Mi']
    
    num_cols = ['Yas', 'Kredi_Skoru']
    cat_cols = ['Sehir', 'Cinsiyet']
    
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    
    X[num_cols] = num_imputer.fit_transform(X[num_cols])
    X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
    
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.30, random_state=42, stratify=y
    )
    
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"{dataset_name} için Başarı Metrikleri Raporu:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"{dataset_name} Doğruluk (Accuracy): {acc:.2f}\n")
    return model, X_encoded, acc

model_kucuk, X_enc_kucuk, acc_kucuk = analiz_yap(df_kucuk.copy(), "Orijinal (Küçük) Veri Seti")


# ==========================================
# 2. YAPAY ZEKA İLE ÇOĞALTILMIŞ VERİ SETİ (200 Kayıt)
# ==========================================
print("--- Bölüm 2: Çoğaltılmış Veri Seti (200 Kayıt) Analizi ---")
np.random.seed(42)
n_samples = 200

veri_sozlugu_buyuk = {
    'Musteri_ID': list(range(1, n_samples + 1)),
    'Yas': np.random.randint(18, 70, size=n_samples).astype(float),
    'Kredi_Skoru': np.random.randint(400, 850, size=n_samples).astype(float),
    'Sehir': np.random.choice(['İstanbul', 'Ankara', 'İzmir', 'Bursa', 'Antalya'], size=n_samples),
    'Cinsiyet': np.random.choice(['Kadın', 'Erkek'], size=n_samples),
    'Terk_Etti_Mi': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]) # %30 terk etme oranı
}

# Rastgele eksik veriler ekleyelim
for i in np.random.choice(n_samples, size=15, replace=False):
    veri_sozlugu_buyuk['Yas'][i] = np.nan
for i in np.random.choice(n_samples, size=10, replace=False):
    veri_sozlugu_buyuk['Kredi_Skoru'][i] = np.nan
for i in np.random.choice(n_samples, size=8, replace=False):
    veri_sozlugu_buyuk['Cinsiyet'][i] = np.nan

df_buyuk = pd.DataFrame(veri_sozlugu_buyuk).set_index('Musteri_ID')
model_buyuk, X_enc_buyuk, acc_buyuk = analiz_yap(df_buyuk.copy(), "Yapay Zeka ile Çoğaltılmış Veri Seti")

print("--- Bölüm 3: Kıyaslama Raporu ---")
print(f"Az kayıtlı (10 adet) veri ile doğruluk oranı: {acc_kucuk:.2f}")
print(f"Çok kayıtlı (200 adet) veri ile doğruluk oranı: {acc_buyuk:.2f}")
print("Yorum: Az kayıtlı veri setlerinde test seti çok küçük olduğu için doğruluk oranı yapay bir şekilde yüksek veya düşük çıkabilir ve genellenebilirliği yoktur. Çoğaltılmış veri setinde modelin doğruluk oranı daha gerçekçidir ve model daha genellenebilir örüntüler öğrenir.")
