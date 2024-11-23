"""
Setelah dilakukan data loading, sekarang kita lakukan data cleaning dan transformation

"""

import pandas as pd
test = pd.read_csv(r"Dataset\test.csv")  # Raw string untuk menghindari masalah escape sequence
train = pd.read_csv(r"Dataset\train.csv")

# ========================================================================================================================

"""
Mengidentifikasi Informasi Dataset
Kita akan melihat informasi dasar tentang dataset, seperti jumlah baris, kolom, tipe data, dan jumlah nilai yang hilang. 
"""

# Menampilkan ringkasan info terkait dataset
# print(train.info())


# Menampilkan statistik deskriptif dari dateset
# print(train.describe(include="all"))


# Memeriksa jumlah nilai yang hilang pada setiap kolom
missing_values = train.isnull().sum()
# print(missing_values[missing_values > 0]) 

"""
Mengatasi Missing Values
"""

# Memisahkan kolom yang memiliki missing values lebih dari 75% dan kurang dari 75%
less = missing_values[missing_values < 1000].index
over = missing_values[missing_values >= 1000].index
# print(less)
# print(over)   


# Contoh mengisi nilai yang hilang dengan median untuk kolom numerik
numeric_features = train[less].select_dtypes(include=['number']).columns
train[numeric_features] = train[numeric_features].fillna(train[numeric_features].median())
# print(train[numeric_features])


# Contoh mengisi nilai yang hilang dengan media untuk kolom kategori
kategorical_features = train[less].select_dtypes(include= ['object']).columns

for column in kategorical_features:
    train[column] = train[column].fillna(train[column].mode()[0])
# print(train[column])


# Menghapus kolom dengan terlalu banyak nilai yang hilang
df = train.drop(columns= over)
# print(df)

# Lakukan pemeriksaan terhadap data yang sudah melewati tahapan verifikasi missing values
missing_values = df.isnull().sum()
# print(missing_values[missing_values > 0])

"""
Mengatasi Outliners
Menggunakan metode IQR (Interquartile Range)
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# for feature in numeric_features:
#     plt.figure(figsize=(10,6))
#     sns.boxplot(x=df[feature])
#     plt.title(f'Box Plot of {feature}')
#     # plt.show()

# Contoh sederhana untuk mengidentifikasi outliners menggunakan IQR
Q1 = df[numeric_features].quantile(0.25)
Q3 = df[numeric_features].quantile(0.75)
IQR = Q3 - Q1

"""
Metode dengan menghapus outliner
"""

# # Filter dataframe untuk hanya menyimpan baris yang tidak mengandung outliners pada kolom numerik
# condition = ~((df[numeric_features] < (Q1 - 1.5 * IQR)) | (df[numeric_features] > (Q3 + 1.5 * IQR))).any(axis=1)
# df_filtered_numeric = df.loc[condition, numeric_features]

# # Menggabungkan kembali dengan kolom kategorikal
# categorical_features = df.select_dtypes(include=['object']).columns
# df = pd.concat([df_filtered_numeric, df.loc[condition, categorical_features]], axis=1)

# for feature in df.columns:
#     plt.figure(figsize=(10,6))
#     sns.boxplot(x=df[feature])
#     plt.title(f'Box Plot of {feature}')
#     plt.show()

"""
Jika Anda tidak ingin menghapus outliers seperti contoh di atas, silakan gunakan metode agregasi seperti berikut.
"""

median = df[numeric_features].median()

# Mengganti nilai outliers dengan median untuk setiap kolom numerik
for feature in numeric_features:
    # Menggunakan lambda untuk mengganti nilai outliers dengan median
    df[feature] = df[feature].apply(lambda x: median[feature] if x < (Q1[feature] - 1.5 * IQR[feature]) or x > (Q3[feature] + 1.5 * IQR[feature]) else x)

# for feature in numeric_features:
#     plt.figure(figsize=(10,6))
#     sns.boxplot(x=df[feature])
#     plt.title(f'Box Plot of {feature}')
#     plt.show()

"""
Normalisasi dan standarisasi data
"""

from sklearn.preprocessing import StandardScaler

# Standarisasi fitur numerik
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Histogram sebelum standarisasi
plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
# sns.histplot(train[numeric_features[3]], kde=True)
# plt.title("Histogram Sebelum Standarisasi")
# plt.show()

# Histogram setelah standarisasi
plt.subplot(1,2,2)
sns.histplot(df[numeric_features[3]], kde=True)
plt.title("Histogram Setelah Standarisasi")
# plt.show()

"""
Menangani duplikasi data
"""

# Mengidentifikasi baris duplikat
duplicates = df.duplicated()

# print("Baris duplikat")
# print(df[duplicates])

# Sayangnya tidak ada duplikasi, tetapi jika kasusnya selain ini dan terdapat duplikasi :
# df = df.drop_duplicates()

# print("Dataframe setelahmenghapus duplikat:")
# print(df)

"""
Mengonversi Tipe Data
Model machine learning tidak dapat langsung menerima input kategorikal.
Hal ini karena mereka bergantung pada operasi matematika yang memerlukan input numerik.
Ada 3 cara untuk menangani hal ini :
    1. One-Hot Encoding
    2. Label Encoding
    3. Ordinal Encoding
"""

# Cek Data kategorikal
category_features = df.select_dtypes(include=['object']).columns
df[category_features]

# One-Hot Encoding
df_one_hot = pd.get_dummies(df, columns=category_features)
print(df_one_hot)

# Label Encoding
from sklearn.preprocessing import LabelEncoder

# Inisialisasi LabelEncoder
label_encoder = LabelEncoder()
df_lencoder = pd. DataFrame(df)

for col in category_features:
    df_lencoder[col] = label_encoder.fit_transform(df[col])

# Menampilkan hasil
print(df_lencoder)