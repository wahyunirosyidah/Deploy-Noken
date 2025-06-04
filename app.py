import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Judul
st.title("Klasifikasi Gambar Menggunakan Model .h5")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('mobilenetv2.h5')
    return model

model = load_model()

# Label klasifikasi
class_names = ['Kucing', 'Anjing', 'Burung']  # Ganti sesuai label kamu

# Upload gambar
uploaded_file = st.file_uploader("Unggah gambar di sini", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # Preprocessing (ganti ukuran dan normalisasi sesuai model)
    img = image.resize((224, 224))  # Sesuaikan dengan input modelmu
    img_array = np.array(img) / 255.0  # Normalisasi
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan batch dimensi

    # Prediksi
    prediction = model.predict(img_array)
    predicted_label = class_names[np.argmax(prediction)]

    # Output hasil
    st.subheader("Hasil Prediksi:")
    st.write(f"Model memprediksi: **{predicted_label}**")
    
    # Tampilkan grafik probabilitas
    st.subheader("Probabilitas Kelas:")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_names[i]}: {prob:.4f}")
