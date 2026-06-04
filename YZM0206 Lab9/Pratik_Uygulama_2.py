import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

print("=== PRATİK UYGULAMA 2 BAŞLADI ===\n")

# ==========================================
# ADIM 1: Veri Setini Yükleme ve Keşifçi Analiz (EDA)
# ==========================================
print("--- Adım 1: Veri Yükleme ve EDA ---")
df = pd.read_csv("Heart attack.csv")

print("Veri Seti İlk 3 Satır:")
print(df.head(3))

print("\nEksik Veri Kontrolü (İşlem Öncesi):")
print(df.isnull().sum())

# Sayısal değişkenler için eksik verileri medyan ile doldurma (Imputation)
# Mevcut veri setinde eksik veri varsa SimpleImputer kullanalım
imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

print("\nEksik Veri Kontrolü (İşlem Sonrası):")
print(df_imputed.isnull().sum())


# ==========================================
# ADIM 2: Önişleme (Preprocessing) ve Dönüşümler
# ==========================================
print("\n--- Adım 2: Önişleme ve Dönüşümler ---")
# Hedef değişken (stroke) ayırma
y = df_imputed['stroke']
X = df_imputed.drop(columns=['stroke'])

# Kategorik olduğu belirtilen öznitelikler
# Veri setinde sayısal olarak gelmiş olsalar bile (0.5, 0.75 vb.), talimat gereği 
# kategorik kabul edip One-Hot Encoding yapıyoruz.
kategorik_kolonlar = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

# get_dummies'in çalışması için bu kolonları önce string'e (kategoriye) çeviriyoruz
X[kategorik_kolonlar] = X[kategorik_kolonlar].astype(str)

# One-Hot Encoding
X_encoded = pd.get_dummies(X, columns=kategorik_kolonlar, drop_first=True)
print("One-Hot Encoding sonrası öznitelik sayısı:", X_encoded.shape[1])


# ==========================================
# ADIM 3: Eğitim ve Test Setlerinin Ayrılması
# ==========================================
print("\n--- Adım 3: Veri Setinin Bölünmesi ---")
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.30, random_state=42, stratify=y
)
print(f"Eğitim Seti: {len(X_train)} kayıt")
print(f"Test Seti: {len(X_test)} kayıt")


# ==========================================
# ADIM 4: Kısıtlamasız Model Eğitimi (Overfitting Gözlemi)
# ==========================================
print("\n--- Adım 4: Kısıtlamasız Model Eğitimi (Overfitting Gözlemi) ---")
model_unconstrained = DecisionTreeClassifier(random_state=42)
model_unconstrained.fit(X_train, y_train)

y_pred_train_unc = model_unconstrained.predict(X_train)
y_pred_test_unc = model_unconstrained.predict(X_test)

acc_train_unc = accuracy_score(y_train, y_pred_train_unc)
acc_test_unc = accuracy_score(y_test, y_pred_test_unc)

print(f"Kısıtlamasız Model Eğitim Doğruluğu: {acc_train_unc:.4f}")
print(f"Kısıtlamasız Model Test Doğruluğu  : {acc_test_unc:.4f}")
print("Yorum: Eğitim doğruluğu %100'e yakınken (veya çok yüksekken) test doğruluğunun daha düşük olması, modelin eğitim verisini ezberlediğini (overfitting/aşırı öğrenme) gösterir.")


# ==========================================
# ADIM 5: Hiperparametre Optimizasyonu (Budama)
# ==========================================
print("\n--- Adım 5: Budanmış Model Eğitimi (max_depth=4) ---")
model_pruned = DecisionTreeClassifier(max_depth=4, random_state=42)
model_pruned.fit(X_train, y_train)

y_pred_train_pruned = model_pruned.predict(X_train)
y_pred_test_pruned = model_pruned.predict(X_test)

acc_train_pruned = accuracy_score(y_train, y_pred_train_pruned)
acc_test_pruned = accuracy_score(y_test, y_pred_test_pruned)

print(f"Budanmış Model Eğitim Doğruluğu: {acc_train_pruned:.4f}")
print(f"Budanmış Model Test Doğruluğu  : {acc_test_pruned:.4f}")
print("Yorum: Maksimum derinlik kısıtlaması (budama) sayesinde eğitim ve test doğrulukları birbirine daha çok yaklaşmış, modelin ezber yapması engellenmiştir.")


# ==========================================
# ADIM 6: Ağaç Yapısının Görselleştirilmesi ve Tıbbi Yorumlama
# ==========================================
print("\n--- Adım 6: Karar Ağacının Görselleştirilmesi ---")
plt.figure(figsize=(15, 8))
plot_tree(
    model_pruned, 
    feature_names=X_encoded.columns.tolist(), 
    class_names=['No Stroke (0)', 'Stroke (1)'], 
    filled=True, 
    rounded=True,
    fontsize=10
)
plt.title("Budanmış Karar Ağacı Yapısı (max_depth=4)")
plt.savefig("Karar_Agaci_Gorseli.png")
print("Karar ağacı görseli 'Karar_Agaci_Gorseli.png' olarak kaydedildi.")

# Kök düğüm tespiti
kok_dugum_index = model_pruned.tree_.feature[0]
kok_dugum_ismi = X_encoded.columns[kok_dugum_index]
print(f"\nKök Düğümde Seçilen Öznitelik: {kok_dugum_ismi}")
print(f"Tıbbi Yorum: Algoritmanın '{kok_dugum_ismi}' özniteliğini en tepeye koyması, inme (stroke) riskini belirlemede Gini safsızlığını en çok azaltan (en yüksek bilgi kazancını sağlayan) faktörün bu olduğunu gösterir. Tıbbi olarak yaş veya kalp rahatsızlığı gibi faktörler inme için genellikle en güçlü risk göstergeleridir.")

print("\n=== PRATİK UYGULAMA 2 TAMAMLANDI ===")
