import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# VERİ SETİNİ YÜKLEME
# ==========================================
veri_sozlugu = {
    'Musteri_ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Yas': [34, 42, np.nan, 23, 55, 40, 28, 48, np.nan, 62],
    'Kredi_Skoru': [650, 580, 710, 620, np.nan, 690, 510, 730, 640, 600],
    'Sehir': ['İstanbul', 'Ankara', 'İstanbul', 'İzmir', 'Ankara', 'İstanbul', 'İzmir', 'Ankara', 'İstanbul', 'İzmir'],
    'Cinsiyet': ['Kadın', 'Erkek', 'Erkek', 'Kadın', np.nan, 'Erkek', 'Kadın', 'Erkek', 'Kadın', 'Erkek'],
    'Terk_Etti_Mi': [0, 1, 0, 0, 1, 0, 1, 1, 0, 1]
}
df = pd.DataFrame(veri_sozlugu).set_index('Musteri_ID')

print("=== SÜREÇ BAŞLADI ===\n")

# ==========================================
# ADIM 1: AÇIKLAYICI VERI ANALİZİ VE GÖRSELLEŞTİRME
# ==========================================
print("--- Adım 1: Veri Seti Yapısı ---")
print(df.info())
print("\nEksik Değer Sayıları:\n", df.isnull().sum())

# Görselleştirme 1: Hedef Değişken Dağılımı
plt.figure(figsize=(5, 4))
sns.countplot(x='Terk_Etti_Mi', data=df, palette='Set2')
plt.title('Adım 1: Hedef Değişken (Terk Etti Mi?) Dağılımı')
plt.xlabel('0: Kalıcı Müşteri | 1: Terk Eden Müşteri')
plt.ylabel('Kişi Sayısı')
plt.show()

# ==========================================
# ADIM 2: EKSİK VERİLERİ TAHMİN ETME / VERI ATAMA
# ==========================================
X = df.drop(columns=['Terk_Etti_Mi'])
y = df['Terk_Etti_Mi']

num_cols = ['Yas', 'Kredi_Skoru']
cat_cols = ['Sehir', 'Cinsiyet']

# Nümeriklere Medyan, Kategoriklere En Çok Tekrar Eden (Mode) Atama
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

X[num_cols] = num_imputer.fit_transform(X[num_cols])
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

print("\n--- Adım 2: Veri Atama Tamamlandı (Eksik Değer Kalmadı) ---")

# ==========================================
# ADIM 3: ONE-HOT ENCODING
# ==========================================
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
print("\n--- Adım 3: Kategorik Veriler Nümeriğe Çevrildi ---")
print(X_encoded.head(3))

# ==========================================
# ADIM 4: VERİ SETİNİ EĞİTİM VE TEST OLARAK AYIRMA
# ==========================================
# Küçük bir veri seti olduğu için test_size=0.30 seçildi
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.30, random_state=42, stratify=y
)
print(f"\n--- Adım 4: Veri Ayrıldı --- Eğitim: {len(X_train)} örnek, Test: {len(X_test)} örnek")

# ==========================================
# ADIM 5: MODELİ EĞİTME VE TAHMİN YAPMA
# ==========================================
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\n--- Adım 5: Karar Ağacı Eğitildi ve Test Verisi Tahmin Edildi ---")

# ==========================================
# ADIM 6: SINIFLANDIRMA BAŞARI METRİKLERİ
# ==========================================
print("\n--- Adım 6: Başarı Metrikleri Raporu ---")
print(classification_report(y_test, y_pred, zero_division=0))
print("Karmaşıklık Matrisi:")
print(confusion_matrix(y_test, y_pred))

# ==========================================
# ADIM 7: KARAR AĞACINI GÖRSELLEŞTİRME
# ==========================================
plt.figure(figsize=(10, 6))
plot_tree(
    model, 
    feature_names=X_encoded.columns.tolist(), 
    class_names=['Kalan (0)', 'Terk (1)'], 
    filled=True, 
    rounded=True,
    fontsize=10
)
plt.title("Adım 7: Modelin Karar Ağacı Ağaç Yapısı")
plt.show()

# ==========================================
# ADIM 8: ÖZNİTELİK SIRALAMASINI GÖRSELLEŞTİRME
# ==========================================
df_importance = pd.DataFrame({
    'Öznitelik': X_encoded.columns,
    'Önem Derecesi': model.feature_importances_
}).sort_values(by='Önem Derecesi', ascending=False)

plt.figure(figsize=(8, 4))
sns.barplot(x='Önem Derecesi', y='Öznitelik', data=df_importance, palette='viridis')
plt.title('Adım 8: Modelin Öznitelik Önem Sıralaması')
plt.xlabel('Önem Skoru')
plt.ylabel('Değişkenler')
plt.tight_layout()
plt.show()

print("\n=== TÜM ADIMLAR BAŞARIYLA TAMAMLANDI ===")