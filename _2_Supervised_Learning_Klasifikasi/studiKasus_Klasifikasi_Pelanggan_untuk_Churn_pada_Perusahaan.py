# Latihan Studi Kasus: Klasifikasi Pelanggan untuk Churn pada Perusahaan XYZ

"""
Dengan menggunakan dataset pelanggan dari perusahaan, kita akan mengimplementasikan beberapa algoritma 
klasifikasi yang telah dipelajari. 

Tujuan utama adalah melatih model-model ini dan mengevaluasi kinerjanya dalam memprediksi kemungkinan 
churn sehingga memberikan wawasan yang berguna untuk tindakan preventif lebih efektif. 
"""

# LANGKAH 1 : MEMUAT LIBRARY
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# LANGKAH 2 : MEMUAT DATA
"""
Pada langkah ini, data yang akan dianalisis diimpor dari Google Drive. Dimulai dengan menentukan ID unik 
file yang diunggah ke Google Drive, ID ini digunakan untuk membuat URL unduhan langsung yang memungkinkan 
akses file CSV melalui kode Python. 
"""

#Gantilah ID file dengan ID dari google drive URL
file_id = '19IfOP0QmCHccMu8A6B2fCUpFqZwCxuzO'

# Buat URL unduhan langsung
download_url = f'https://drive.google.com/uc?id={file_id}'

# Baca file CSV dari URL
data = pd.read_csv(download_url)

# Mengatur pandas untuk menampilkan seluruh kolom 
pd.set_option('display.max_columns', None)

# Tampilkan dataframe untuk memastikan telah dibaca dengan benar
# print(data.head())

# Tampilkan informasi umum tentang dataset
# print("\nInformasi Dataset: ")
# data.info()

# Cek Missing Value
# print("\nMissing values per fitur: ")
# print(data.isnull().sum())

# Hapus kolom 'RowNumber', 'CustomerId', dan 'Surname'
data = data.drop(columns=['RowNumber', 'CustomerId','Surname'])

# Tampilkan DataFrame untuk memastikan kolom telah dihapus
# print(data.head())

# LANGKAH 3 : EXPLORATORY DATA ANALYSIS (EDA)
"""
Pada tahap ini, distribusi fitur numberik pertama-tama dianalisis. Setiap fitur numerik divisualisasikan
menggunakan histogram yang menunjukkan distribusi nilai2 dalam fitur tersebut. Histogram ini dilengkapi
dengan kurva densitas untuk memberikan gambaran lebih jelas tetang pola distribusi data : apakah data
terdistribusi normal atau mengalami skewness?

"""

# Distribusi fitur numerik
num_features = data.select_dtypes(include=[np.number])
plt.figure(figsize=(14, 10))
for i, column in enumerate(num_features.columns, 1):
    plt.subplot(3,4,i)
    sns.histplot(data[column], bins=30, kde=True, color='blue')
    plt.title(f'Distribusi {column}')
plt.tight_layout()
# plt.show()

# Distribusi fitur kategorikal
cat_features = data.select_dtypes(include=[object])
plt.figure(figsize=(14, 8))
for i, column in enumerate(cat_features.columns, 1):
    plt.subplot(2, 4, i)
    # sns.countplot(y=data[column], palette='viridis')
    sns.countplot(y=data[column], palette='viridis', hue=data[column], legend=False)
    plt.title(f'Distribusi {column}')
plt.tight_layout()
# plt.show()

# Heatmap korelasi untuk fitur numerik
plt.figure(figsize=(12, 10))
correlation_matrix = num_features.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Heatmap Korelasi')
# plt.show()

# Pairplot untuk fitur numerik
sns.pairplot(num_features)
# plt.show()

# Visualisasi distribusi variabel target
plt.figure(figsize=(8, 4))
# sns.countplot(x='Exited', data=data, palette='viridis')
sns.countplot(x='Exited', data=data, palette='viridis', hue=data['Exited'], legend=False)
plt.title('Distribusi Variabel Target (Exited)')
# plt.show()

# LANGKAH 4 : LABEL ENCODER
"""
Pada langkah ini, encoding diterapkan pada fitur kategorikal dalam dataset untuk mempersiapkan data bagi
algoritma pembelajaran mesin. Label Encoder digunakan untuk mengonversi nilai kategorikal menjadi
format numerik yang dapat diproses oleh model
"""

# Buat instance LabelEncoder 
label_encoder = LabelEncoder()

# List kolom kategorikal yang perlu di-encode
categorical_column = ['Geography', 'Gender']

# Encode kolom kategorikal
for column in categorical_column:
    data[column] = label_encoder.fit_transform(data[column])

# Tampilkan Dataframe untuk memastikan encoding telah diterapkan
# print(data.head())

# LANGKAH 5 : DATA SPLITTING
"""
Pada langkah ini, data numerik dinormalisasi menggunakan MinMaxScalar untuk memastikan bahwa
semua fitur numerik berada dalam rentang yang sama, yang dapat meningkatkan performa model.
Setelah normalisasi, data dibagi menjadi fitur (X) dan target (Y)

Data kemudian dipisahkan menjadi set pelatihan dan set uji menggunkan train_test_split
"""

# Buat instance MinMaxScalar
scalar = MinMaxScaler()

# Normalisasi semua kolom numerik
numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
data[numeric_columns] = scalar.fit_transform(data[numeric_columns])

# Pisahkan fitur (X) dan target (y)
X = data.drop(columns=['Exited'])
y = data['Exited']

# Split data menjadi set pelatihan dan set uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tampilkan bentuk set pelatihan dan set uji untuk memastikan split
print(f"Training set shape: X_train = {X_train.shape}, y_train = {y_train.shape}")
print(f"Test set shape: X_test = {X_test.shape}, y_test = {y_test.shape}")

# LANGKAH 6 : PELATIHAN MODEL
"""
Pada langkah ini, setiap algoritma klasifikasi dilatih secara terpisah dengan menggunakan data pelatihan.
Model KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier, SVC, dan GaussianNB 
dipersiapkan serta dilatih. Setelah proses pelatihan selesai, model-model ini siap untuk diuji 
dengan data uji. Pesan "Model training selesai." menandakan bahwa semua model sudah
berhasil dilatih.
"""

# Definisikan setiap klasifikasi secara terpisah
knn = KNeighborsClassifier().fit(X_train, y_train)
dt = DecisionTreeClassifier().fit(X_train, y_train)
rf = RandomForestClassifier().fit(X_train, y_train)
svm = SVC().fit(X_train, y_train)
nb = GaussianNB().fit(X_train, y_train)

print("Model Training Selesai.")

# LANGKAH 7 : EVALUASI MODEL
"""

"""