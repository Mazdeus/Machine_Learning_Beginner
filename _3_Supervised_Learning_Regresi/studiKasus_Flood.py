import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

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

# print("Baris duplikat: ")
# print(df[duplicate])

# EKSPLORATORY DAN EXPLANATORY DATA
# df.describe(include='all')

# Visualisasikan
# Menghitung jumlah variabel
num_vars = df.shape[1]

# Menentukan jumlah baris dan kolom untuk grid subplot
n_cols = 4 # Jumlah kolom yang diinginkan
n_rows = -(-num_vars // n_cols) # Ceiling division untuk menentukan jumlah baris

# Membuat subplot
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))

# Flatten axes array untuk memudahkan iterasi jika diperlukan
axes = axes.flatten()

# Plot setiap variabel
for i, column in enumerate(df.columns):
    df[column].hist(ax=axes[i], bins=20, edgecolor='black')
    axes[i].set_title(column)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')

# Menghapus subplot yang tidak terpakai (jika ada)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Menyesuaikan layout agar lebih rapi
plt.tight_layout()
# plt.show()

# Memilih fitur yang memiliki hubungan dengan fitur target (FloodProbability)
# Menghitung korelasi antar variabel target dan semua variabel lainnya
target_coor = df.corr()['FloodProbability']

# (Opsional) Mengurutkan hasil korelasi berdasarkan kekuatan korelasi
target_coor_sorted = target_coor.abs().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
target_coor_sorted.plot(kind='bar')
plt.title(f'Correlation with FloodProbability')
plt.xlabel('Variables')
plt.ylabel('Correlation Coefficient')
# plt.show()

# SPLITTING DATA
# Memisahkan fitur (X) dan target (y)
X = df.drop(columns=['FloodProbability'])
y = df['FloodProbability']

from sklearn.model_selection import train_test_split

# Membagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Menghitung panjang/jumlah data
print("Jumlah data: ", len(X))
# Menghitung panjang/jumlah data pada X_train
print("Jumlah data train: ", len(X_train))
# Menghitung panjang/jumlah data pada X_test
print("Jumlah data test: ", len(X_test))

# DATA MODELLING
# ALGORITMA LARS
from sklearn import linear_model
lars = linear_model.Lars(n_nonzero_coefs=1).fit(X_train, y_train)

pred_lars = lars.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae_lars = mean_absolute_error(y_test, pred_lars)
mse_lars = mean_squared_error(y_test, pred_lars)
r2_lars = r2_score(y_test, pred_lars)

# print(f"MAE: {mae_lars}")
# print(f"MSE: {mse_lars}")
# print(f"R2: {r2_lars}")

# Membuat dictionary untuk menyimpan hasil evaluasi
data = {
    'MAE': [mae_lars],
    'MSE': [mse_lars],
    'R2': [r2_lars]
}

# Konversi dictionary ke DataFrame
df_results = pd.DataFrame(data, index=['LARS'])
df_results

# ALGORITMA Linear Regression
from sklearn.linear_model import LinearRegression
LR = LinearRegression().fit(X_train, y_train)

pred_LR = LR.predict(X_test)

mae_LR = mean_absolute_error(y_test, pred_LR)
mse_LR = mean_squared_error(y_test, pred_LR)
r2_LR = r2_score(y_test, pred_LR)

# print(f"MAE: {mae_LR}")
# print(f"MSE: {mse_LR}")
# print(f"R2: {r2_LR}")

df_results.loc['Linear Regression'] = [mae_LR, mse_LR, r2_LR]
df_results

# ALGORITMA GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor

GBR = GradientBoostingRegressor(random_state=184)
GBR.fit(X_train, y_train)

pred_GBR = GBR.predict(X_test)

mae_GBR = mean_absolute_error(y_test, pred_GBR)
mse_GBR = mean_squared_error(y_test, pred_GBR)
r2_GBR = r2_score(y_test, pred_GBR)

# print(f"MAE: {mae_GBR}")
# print(f"MSE: {mse_GBR}")
# print(f"R2: {r2_GBR}")

df_results.loc['GradientBoostingRegressor'] = [mae_GBR, mse_GBR, r2_GBR]
df_results