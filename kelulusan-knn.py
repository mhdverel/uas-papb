import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# baca file dan lihat 5 data teratas
data = pd.read_csv("dataset_kelulusan.csv", sep=";")
print("5 Data Teratas")
print(data.head()) 


# melihat tipe data
print("Tipe Data")
print(data.info())

# menentukan variable independen
x = data.drop(["waktu_lulus"], axis=1)
print(x.head())

# menentukan variable dependen
y = data["waktu_lulus"]
print("Variable Independen")
print(y.head())

# bagi data training dan testing
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20)

# mengubah skala data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test  = scaler.transform(x_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)
print("Hasil Prediksi")
print(y_pred)

# ukur keakuratan
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

