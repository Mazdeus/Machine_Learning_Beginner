from os import name
from flask import Flask, request, jsonify
import joblib
 
# Inisialisasi aplikasi Flask
app = Flask(__name__)
 
# Memuat model yang telah disimpan
joblib_model = joblib.load(r'D:\COURSE\DICODING\Machine-Learning\3_Belajar_Machine_Learning_untuk_Pemula\Machine_Learning_Pemula\gbr_model.joblib') # Pastikan path file sesuai dengan penyimpanan Anda
 
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']  # Mengambil data dari request JSON
    prediction = joblib_model.predict(data)  # Melakukan prediksi (harus dalam bentuk 2D array)
    return jsonify({'prediction': prediction.tolist()})
 
if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)

"""
Untuk menjalankannya siapkan 2 terminal
Terminal 1 :
- pastikan versi python dan flask sama
- jalankan : C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe testing_deploy.py
- jangan lupa cd ke direktori tempat testing_deploy.py berada

Terminal 2 :
- Jalankan : 
Invoke-WebRequest -Uri http://127.0.0.1:5000/predict -Method POST -ContentType "application/json" -Body (Get-Content -Raw -Path "D:\COURSE\DICODING\Machine-Learning\3_Belajar_Machine_Learning_untuk_Pemula\Machine_Learning_Pemula\Machine_Learning_Workflow\data.json")

atau

curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d @data.json

atau 

curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d (Get-Content -Raw -Path "D:\COURSE\DICODING\Machine-Learning\3_Belajar_Machine_Learning_untuk_Pemula\Machine_Learning_Pemula\Machine_Learning_Workflow\data.json")
"""

"""
Output :
StatusCode        : 200
StatusDescription : OK
Content           : {
                      "prediction": [
                        1.17380402260657
                      ]
                    }
"""