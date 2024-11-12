# Latihan Studi Kasus: Klasifikasi Pelanggan untuk Churn pada Perusahaan XYZ

"""
Dengan menggunakan dataset pelanggan dari perusahaan, kita akan mengimplementasikan beberapa algoritma 
klasifikasi yang telah dipelajari. 

Tujuan utama adalah melatih model-model ini dan mengevaluasi kinerjanya dalam memprediksi kemungkinan 
churn sehingga memberikan wawasan yang berguna untuk tindakan preventif lebih efektif. 
"""

# Langka 1 : Mengimpor Library
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

# Langkah 2 : Memuat Data
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
print("\nInformasi Dataset: ")
data.info()

# Cek Missing Value
print("\nMissing values per fitur: ")
print(data.isnull().sum())