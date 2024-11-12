"""
Pada Dataset test.csv dan train.csv kita lakukan data loading
Yaitu memasukkan data ke dalam lingkungan pemrograman

"""

import pandas as pd
test = pd.read_csv(r"Dataset\test.csv")  # Raw string untuk menghindari masalah escape sequence
print(test.head())


train = pd.read_csv(r"Dataset\train.csv")
print(train.head())


