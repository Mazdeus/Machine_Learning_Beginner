import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# DATA LOADING

df_train = pd.read_csv(r"Dataset\Flood\train.csv")
# print(df_train)

df_test = pd.read_csv(r"Dataset\Flood\test.csv")
# print(df_test)

# DATA CLEANING DAN TRANSFORMATION

# Menampilkan ringkasan informasi dari dataset
# df_train.info()

# Menampilkan statistik deskriptif dari dataset
# print(df_train.describe(include='all'))

# Pemeriksaan missing value
missing_values = df_train.isnull().sum()
# print(missing_values[missing_values > 0])

# Periksa outliners
# for feature in df_train.columns:
#     plt.figure(figsize=(10, 5))
#     sns.boxplot(x=df_train[feature])
#     plt.title(f'Box plot of {feature}')
#     plt.show()

# Menghapus outliners

# Contoh sederhana untuk mengidentifikasi outliners menggunakan IQR
Q1 = df_train.quantile(0.25)
Q3 = df_train.quantile(0.75)
IQR = Q3 - Q1

# Filter dataframe untuk hanya menyimpan baris yang tidak memiliki outliners pada kolom numerik
condition = ~((df_train < (Q1 - 1.5 * IQR)) | (df_train > (Q3 + 1.5 * IQR))).any(axis=1)
df = df_train.loc[condition]

# Cek outliners
# for feature in df.columns:
#     plt.figure(figsize=(10, 5))
#     sns.boxplot(x=df[feature])
#     plt.title(f'Box plot of {feature}')
#     plt.show()

# Standarisasi pada data
# Memastikan hanya data dengan tipe numerik yang akan di proses
numeric_features = df.select_dtypes(include=['number']).columns

# Plot histogram sebelum standarisasi
# for feature in df_before_scaling.select_dtypes(include=['number']).columns:
#     plt.figure(figsize=(10, 5))
#     sns.histplot(df_before_scaling[feature], kde=True)
#     plt.title(f'Histogram of {feature} before scaling')
#     plt.show()

# Standarisasi fitur numerik
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Plot histogram setelah standarisasi
# for feature in df[numeric_features].columns:
#     plt.figure(figsize=(10, 5))
#     sns.histplot(df[feature], kde=True)
#     plt.title(f'Histogram of {feature} after scaling')
#     plt.show()

# Mengidentifikasi baris duplikat
duplicate = df.duplicated()

print("Baris duplikat: ")
print(df[duplicate])

# EKSPLORATORY DAN EXPLANATORY DATA
df.describe(include='all')

# Visualisasikan
