from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os
import tensorflow as tf

# Membuat instance Flask
app = Flask(__name__)

# Memuat model InceptionV3 yang sudah disimpan
model_path = 'Dermata_inceptionV3_V3.keras'  # Ganti dengan lokasi model
model = load_model(model_path)

# Daftar label untuk model multilabel
labels = [
    "Acne", "Blackheads", "Dark Spots", "Dry Skin",
    "Eye Bags", "Normal Skin", "Oily Skin", "Pores",
    "Redness", "Wrinkles"
]

# Fungsi untuk memproses gambar
def process_image(img_path, target_size=(540, 540, 3)):
    img = image.load_img(img_path, target_size=target_size[:2])  # Sesuaikan ukuran tanpa channel
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch
    img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)  # Preprocessing khusus InceptionV3
    return img_array

# Endpoint untuk prediksi gambar
@app.route('/predict', methods=['POST'])
def predict():
    # Cek apakah gambar ada dalam request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Simpan file gambar sementara
        img_path = os.path.join('uploads', file.filename)
        file.save(img_path)

        # Proses gambar dan buat prediksi
        img_array = process_image(img_path)
        predictions = model.predict(img_array)

        # Menentukan label yang diprediksi berdasarkan ambang batas 0.2 untuk multilabel
        threshold = 0.2
        predicted_labels = (predictions >= threshold).astype(int)

        # Menggabungkan label yang diprediksi dengan koma, serta persentase
        result = []
        for i in range(len(labels)):
            if predicted_labels[0][i] == 1:
                result.append(f"{labels[i]} ({predictions[0][i] * 100:.2f}%)")

        # Jika tidak ada label yang terdeteksi
        if not result:
            result.append("Tidak ada masalah kulit yang terdeteksi.")
        
        # Kembalikan hasil prediksi sebagai response JSON
        return jsonify({'predictions': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Endpoint untuk URL root
@app.route('/')
def home():
    return "Welcome to the Skin Problem Prediction API! Use the '/predict' endpoint to upload an image."

# Endpoint untuk favicon.ico
@app.route('/favicon.ico')
def favicon():
    return '', 204

# Menjalankan aplikasi Flask
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
