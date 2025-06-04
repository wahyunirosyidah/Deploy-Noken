import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort

# ✅ Load ONNX model
@st.cache_resource
def load_model():
    session = ort.InferenceSession("modelkamu.onnx")
    return session

model = load_model()

# ✅ Daftar label
class_names = ['Bitu Agia', 'Junum Ese']

# ✅ Upload gambar
uploaded_file = st.file_uploader("Unggah gambar di sini", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # ✅ Preprocessing gambar
    img = image.resize((224, 224))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ✅ Prediksi ONNX
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    prediction = model.run([output_name], {input_name: img_array})[0]

    predicted_label = class_names[np.argmax(prediction)]

    # ✅ Hasil prediksi
    st.subheader("Hasil Prediksi:")
    st.write(f"Model memprediksi: **{predicted_label}**")

    st.subheader("Probabilitas:")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_names[i]}: {prob:.4f}")
