import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model.h5')
    return model

model = load_model()

# ✅ Pastikan ini adalah list dua elemen, bukan satu string
class_names = ['Bitu Agia', 'Junum Ese']

# Upload gambar
uploaded_file = st.file_uploader("Unggah gambar di sini", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # ✅ Ubah ke RGB untuk pastikan channel = 3
    image = image.convert("RGB")
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_label = class_names[np.argmax(prediction)]

    st.subheader("Hasil Prediksi:")
    st.write(f"Model memprediksi: **{predicted_label}**")

    st.subheader("Probabilitas:")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_names[i]}: {prob:.4f}")
