"""
Pada Dataset test.csv dan train.csv kita lakukan data loading
Yaitu memasukkan data ke dalam lingkungan pemrograman

"""

import pandas as pd
# test = pd.read_csv(r"Dataset\test.csv")  # Raw string untuk menghindari masalah escape sequence
# print(test.head())


train = pd.read_csv(r"Dataset\train.csv")
# print(train.head())


# Menampilkan ringkasan info terkait dataset
# print(train.info())


# Menampilkan statistik deskriptif dari dateset
# print(train.describe(include="all"))


# Memeriksa jumlah nilai yang hilang pada setiap kolom
missing_values = train.isnull().sum()
# print(missing_values[missing_values > 0]) 


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
print(missing_values[missing_values > 0])