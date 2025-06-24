# Gerekli kütüphaneler
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay,
    precision_recall_curve, PrecisionRecallDisplay
)
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score

# Uyarıları görmezden gel
warnings.filterwarnings("ignore")

# Veri setini oku
df = pd.read_csv("C:/Users/atanm/OneDrive/Masaüstü/ai4i2020.csv")

# İlk bakış
print(df.info())
print(df.isnull().sum())
print(f"Duplicated rows: {df.duplicated().sum()}")
print(df['Type'].value_counts())

# Kullanılmayan sütunları kaldır
df.drop(columns=['UDI', 'Product ID'], inplace=True)

# Yeni özellikler oluştur
df['temperature_difference'] = df['Process temperature [K]'] - df['Air temperature [K]']
df['Mechanical Power [W]'] = np.round((df['Torque [Nm]'] * df['Rotational speed [rpm]'] * 2 * np.pi) / 60, 4)

# Veri görselleştirme
plt.figure(figsize=(6, 4))
sns.countplot(x='Type', data=df)
plt.title('Distribution of Machine Types')
plt.xlabel('Machine Type')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='Type', hue='Machine failure', data=df)
plt.title('Machine Failure Distribution Across Types')
plt.xlabel('Machine Type')
plt.ylabel('Count')
plt.legend(title='Failure')
plt.show()

# Özelliklerin dağılımı ve aykırı değer kontrolü
cols = ['Torque [Nm]', 'Rotational speed [rpm]', 'temperature_difference', 'Mechanical Power [W]']
for col in cols:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(data=df, x=col, kde=True, ax=axes[0])
    axes[0].set_title(f"{col} Distribution")
    sns.boxplot(data=df, x=col, ax=axes[1])
    axes[1].set_title(f"{col} - Outlier Check")
    plt.tight_layout()
    plt.show()

# Korelasyon analizi
sns.pairplot(df[['Torque [Nm]', 'Rotational speed [rpm]', 'temperature_difference','Mechanical Power [W]', 'Machine failure']], hue='Machine failure')
plt.show()

corr_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(8, 5))
sns.heatmap(corr_matrix, annot=True, cmap='cividis', fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.show()

# Arızalar arasındaki korelasyon
target = df.iloc[:, [6, 7, 8, 9, 10, 11]]  # TWF, HDF, PWF, OSF, RNF, Machine failure
target_mat = target.corr()
sns.heatmap(target_mat, annot=True, cmap="cividis", fmt=".4f", linewidth=0.5)
plt.title("Failure Type Correlation Matrix")
plt.show()

# Alt arıza sütunlarını kaldır
df.drop(columns=['TWF', 'HDF', 'PWF', 'OSF', 'RNF'], inplace=True)

# Label encoding
le = LabelEncoder()
df['Type'] = le.fit_transform(df['Type'])

# Hedef değişken ve özelliklerin ayrılması
Y = df['Machine failure'].copy()
X = df.drop('Machine failure', axis=1)

# Feature Scaling - DÜZELTME BURADA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

# Sınıf dengesizliğini kontrol et
print("Original class distribution:", Counter(Y))

# SMOTE ile dengeleme
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, Y)
print("Balanced class distribution:", Counter(y_resampled))

# Eğitim/test veri bölme
X_train, X_test, Y_train, Y_test = train_test_split(X_resampled, y_resampled, test_size=0.1, random_state=42)

# Modellerin tanımlanması
models = {
    'Logistic Regression': LogisticRegression(),
    'Logistic Regression CV': LogisticRegressionCV(),
    'SGD': SGDClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Bagging': BaggingClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Support Vector Machine': SVC(probability=True),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

# Cross-validation ile model değerlendirme
def evaluate_model_cross_val(X, Y, cv=5):
    result = []
    for name, model in models.items():
        scores = cross_val_score(model, X, Y, cv=cv, scoring='accuracy')
        mean_score = scores.mean()
        result.append((name, mean_score))
    result.sort(key=lambda x: x[1], reverse=True)
    return result

# Performans karşılaştırması
results_cv = evaluate_model_cross_val(X_resampled, y_resampled, cv=5)

# Sonuçları yazdır
print("Cross-Validation Model Performance:")
for name, acc in results_cv:
    print(f"{name}: {acc:.6f}")

# Doğrulukların görselleştirilmesi (bar plot)
def plot_model_accuracies(results):
    names = [name for name, acc in results]
    accuracies = [acc for name, acc in results]

    plt.figure(figsize=(12, 6))
    sns.barplot(x=accuracies, y=names, palette="viridis")
    plt.xlabel("Ortalama Doğruluk")
    plt.ylabel("Modeller")
    plt.title("Modellerin Ortalama Doğruluk Karşılaştırması (Cross-Validation ile)")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.show()

# Grafik çizdir
plot_model_accuracies(results_cv)

# Değerlendirme fonksiyonu
def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"--------- {model_name} Classification Report ------\n")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f"{model_name} - Confusion Matrix")
    plt.show()

    # ROC Curve
    roc_auc = roc_auc_score(y_test, y_prob)
    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title(f"{model_name} - ROC Curve (AUC = {roc_auc:.2f})")
    plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    PrecisionRecallDisplay(precision=precision, recall=recall).plot()
    plt.title(f"{model_name} - Precision-Recall Curve")
    plt.show()

# En iyi model (Random Forest) ile değerlendirme
RF = RandomForestClassifier(class_weight='balanced', random_state=42)
RF.fit(X_train, Y_train)
evaluate_model(RF, X_test, Y_test, model_name="Random Forest")

# KNN modeli eğitimi ve değerlendirmesi - GridSearchCV ile
print("\n=== KNN Model Optimizasyonu ===")

# GridSearchCV için parametre ızgarası
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

# Base KNN modelini oluştur
KNN = KNeighborsClassifier()

# GridSearchCV ile optimizasyon
grid_search = GridSearchCV(
    estimator=KNN,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Grid search'ü çalıştır
print("\nKNN için GridSearchCV optimizasyonu başlatılıyor...")
grid_search.fit(X_train, Y_train)

# En iyi sonuçları yazdır
print("\nEn iyi parametreler:", grid_search.best_params_)
print("En iyi cross-validation skoru: {:.4f}".format(grid_search.best_score_))

# En iyi KNN modelini al ve değerlendir
best_knn = grid_search.best_estimator_
evaluate_model(best_knn, X_test, Y_test, model_name="Optimize Edilmiş KNN")

# GridSearch sonuçlarını görselleştir
plt.figure(figsize=(10, 6))
results = pd.DataFrame(grid_search.cv_results_)
scores = results.groupby('param_n_neighbors')['mean_test_score'].mean()
plt.plot(scores.index, scores.values, marker='o')
plt.title('K Değerinin Model Performansına Etkisi')
plt.xlabel('K Değeri')
plt.ylabel('Ortalama Test Skoru')
plt.grid(True)
plt.show()

# Modeli kaydet
with open("model.pkl", "wb") as f:
    pickle.dump(RF, f)

# Scaler'ı da kaydet
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Yeni eklenen interaktif tahmin fonksiyonu
def get_failure_explanation(features, probability):
    """Arıza olasılığına göre açıklama üreten fonksiyon"""
    high_temp_diff = features['temperature_difference'] > X['temperature_difference'].mean()
    high_torque = features['Torque [Nm]'] > X['Torque [Nm]'].mean()
    high_speed = features['Rotational speed [rpm]'] > X['Rotational speed [rpm]'].mean()
    high_power = features['Mechanical Power [W]'] > X['Mechanical Power [W]'].mean()
    high_tool_wear = features['Tool wear [min]'] > X['Tool wear [min]'].mean()
    
    explanation = []
    
    if probability >= 0.7:
        explanation.append("⚠️ YÜKSEK arıza riski tespit edildi!")
        if high_temp_diff:
            explanation.append("- Sıcaklık farkı normalin üzerinde, bu termal strese işaret edebilir.")
        if high_torque:
            explanation.append("- Yüksek tork değeri mekanik zorlanmaya neden olabilir.")
        if high_speed:
            explanation.append("- Yüksek dönüş hızı aşınmayı hızlandırabilir.")
        if high_power:
            explanation.append("- Yüksek mekanik güç, sistemin zorlandığını gösteriyor.")
        if high_tool_wear:
            explanation.append("- Takım aşınması kritik seviyede, değiştirilmesi gerekebilir.")
    
    elif probability >= 0.4:
        explanation.append("⚠️ ORTA seviye arıza riski tespit edildi.")
        if high_temp_diff or high_torque:
            explanation.append("- Sıcaklık ve/veya tork değerleri dikkat edilmesi gereken seviyelerde.")
        if high_speed or high_power:
            explanation.append("- Çalışma parametreleri optimize edilebilir.")
        if high_tool_wear:
            explanation.append("- Takım aşınması takip edilmeli.")
    
    else:
        explanation.append("✅ DÜŞÜK arıza riski tespit edildi.")
        explanation.append("- Makine normal parametreler dahilinde çalışıyor.")
        if any([high_temp_diff, high_torque, high_speed, high_power, high_tool_wear]):
            explanation.append("- Bazı parametreler yüksek olsa da, genel durum stabil görünüyor.")
    
    return "\n".join(explanation)

def predict_failure():
    """Kullanıcıdan veri alıp tahmin yapan interaktif fonksiyon"""
    print("\n=== Makine Arıza Tahmini ===")
    
    # Kullanıcıdan veri alma
    machine_types = {
        'L': 0,  # Low
        'M': 1,  # Medium
        'H': 2   # High
    }
    
    print("\nMakine Tipini Seçin:")
    print("L: Düşük Kapasite")
    print("M: Orta Kapasite")
    print("H: Yüksek Kapasite")
    
    while True:
        machine_type = input("Makine Tipi (L/M/H): ").upper()
        if machine_type in machine_types:
            break
        print("Geçersiz giriş! Lütfen L, M veya H girin.")
    
    # Diğer özellikleri al
    air_temp = float(input("\nHava Sıcaklığı [K]: "))
    process_temp = float(input("İşlem Sıcaklığı [K]: "))
    rot_speed = float(input("Dönüş Hızı [rpm]: "))
    torque = float(input("Tork [Nm]: "))
    tool_wear = float(input("Takım Aşınması [min]: "))
    
    # Özellik hesaplamaları
    temp_diff = process_temp - air_temp
    mech_power = np.round((torque * rot_speed * 2 * np.pi) / 60, 4)
    
    # Veriyi modelin beklediği formata dönüştür
    features = pd.DataFrame({
        'Type': [machine_types[machine_type]],
        'Air temperature [K]': [air_temp],
        'Process temperature [K]': [process_temp],
        'Rotational speed [rpm]': [rot_speed],
        'Torque [Nm]': [torque],
        'Tool wear [min]': [tool_wear],
        'temperature_difference': [temp_diff],
        'Mechanical Power [W]': [mech_power]
    })
    
    # Veriyi ölçeklendir
    scaled_features = scaler.transform(features)
    
    # Tahmin yap
    failure_prob = RF.predict_proba(scaled_features)[0][1]
    
    # Sonuçları göster
    print("\n=== Tahmin Sonuçları ===")
    print(f"\nArıza Olasılığı: %{failure_prob*100:.2f}")
    
    # Detaylı açıklama
    explanation = get_failure_explanation(features.iloc[0], failure_prob)
    print("\nDetaylı Analiz:")
    print(explanation)
    
    # Önemli parametreleri göster
    print("\nÖlçülen Parametreler:")
    print(f"Sıcaklık Farkı: {temp_diff:.2f} K")
    print(f"Mekanik Güç: {mech_power:.2f} W")
    print(f"Takım Aşınması: {tool_wear:.0f} min")

# Ana döngü - programı başlat
while True:
    response = input("\nYeni bir tahmin yapmak ister misiniz? (E/H): ").upper()
    if response == 'E':
        predict_failure()
    elif response == 'H':
        print("Program sonlandırılıyor...")
        break
    else:
        print("Geçersiz giriş! Lütfen E veya H girin.") 