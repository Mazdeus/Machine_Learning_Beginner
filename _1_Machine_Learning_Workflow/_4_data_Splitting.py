"""
Setelah dilakukan data loading, sekarang kita lakukan data cleaning dan transformation

"""

import pandas as pd
test = pd.read_csv(r"Dataset\test.csv")  # Raw string untuk menghindari masalah escape sequence
train = pd.read_csv(r"Dataset\train.csv")

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

# for feature in numeric_features:
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
# print(df_one_hot)

# Label Encoding
from sklearn.preprocessing import LabelEncoder

# Inisialisasi LabelEncoder
label_encoder = LabelEncoder()
df_lencoder = pd. DataFrame(df)

for col in category_features:
    df_lencoder[col] = label_encoder.fit_transform(df[col])

# Menampilkan hasil
# print(df_lencoder)


# =======================================================================================

"""
Eksploratory dan eksplanatory data analysis
"""

#Pemeriksaan kembali terhadap dataframe yang kita gunakan
# print(df_lencoder.head())

""" 
Memeriksa kembali missing values
"""

# menghitung jumlah dan persentase missing values di setiap kolom
missing_values = df_lencoder.isnull().sum()
missing_percentage = (missing_values / len(df_lencoder)) * 100

missing_data = pd.DataFrame({
    'Missing Values' : missing_values,
    'Percentage' : missing_percentage
}).sort_values(by='Missing Values', ascending=False)

# print(missing_data[missing_data['Missing Values'] > 0]) # Menampilkan kolom dengan missing values

"""
Melihat sebaran data dengan histogram
"""

# Menghitung jumlah variabel
num_vars = df_lencoder.shape[1]
 
# Menentukan jumlah baris dan kolom untuk grid subplot
n_cols = 4  # Jumlah kolom yang diinginkan
n_rows = -(-num_vars // n_cols)  # Ceiling division untuk menentukan jumlah baris
 
# Membuat subplot
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
 
# Flatten axes array untuk memudahkan iterasi jika diperlukan
axes = axes.flatten()
 
# Plot setiap variabel
for i, column in enumerate(df_lencoder.columns):
    df_lencoder[column].hist(ax=axes[i], bins=20, edgecolor='black')
    axes[i].set_title(column)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
 
# Menghapus subplot yang tidak terpakai (jika ada)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
 
# Menyesuaikan layout agar lebih rapi
# plt.tight_layout()
# plt.show()

"""
Visualisasikan Distribusi beberapa kolom serta melihat korelasi 
antara variabel numerik
"""

import seaborn as sns
import matplotlib.pyplot as plt

# Visualisasi distribusi data untuk beberapa kolom
columns_to_plot = ['OverallQual', 'YearBuilt', 'LotArea', 'SaleType', 'SaleCondition']

plt.figure(figsize=(15,10))
for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df_lencoder[column], kde=True, bins=30)
    plt.title(f'Distribution of {column}')

# plt.tight_layout()
# plt.show()

"""
Analisis korelasi dengan membuat matriks korelasi
"""

# Visualisasi korelasi antar variabel numerik
plt.figure(figsize=(12, 10))
correlation_matrix = df_lencoder.corr()

sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
# plt.show()

"""
Matriks korelasi yang hanya terhadap variabel atau fitur SalePrice saja
"""

# Menghitung korelasi antara variabel target dan semua variabel lainnya
target_corr = df_lencoder.corr()['SalePrice']

# (Opsional) Mengurutkan hasil korelasi berdasarkan korelasi
target_corr_sorted = target_corr.abs().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
target_corr_sorted.plot(kind='bar')
plt.title(f'Correlation with SalePrice')
plt.xlabel('Variables')
plt.ylabel('Correlation Coefficient')
# plt.show()

# ==============================================================================

"""
Data splitting
Kita akan membagi dataset menggunakan fungsi train test split (holdup method)
dari library SKLearn
"""

import sklearn

# Memisahkan fitur (x) dan target (y)
x = df_lencoder.drop(columns=['SalePrice'])
y = df_lencoder['SalePrice']

from sklearn.model_selection import train_test_split

# Membagi dataset menjaid training dan testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)


# menghitung panjang/jumlah data
print("Jumlah data: ",len(x))

# Menghitung panjang/jumlah data pada x_test
print("Jumlah data latih: ",len(x_train))

# Menghitung panjang/jumlah data pada x_test
print("Jumlah data test: ",len(x_test))

