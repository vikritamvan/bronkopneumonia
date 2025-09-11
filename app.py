import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Konfigurasi halaman
st.set_page_config(
    page_title="STAT-RESPIRA",
    page_icon="🔍",
    layout="centered"
)

# Navigasi Sidebar
st.sidebar.title("Navigasi")
pilihan = st.sidebar.radio("Pilih Halaman", ["Beranda", "Deteksi"])

# --- Bagian Halaman Beranda ---
if pilihan == "Beranda":
    # Logo + Judul
    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px; margin-right: 10px;">🩺</span>
            <h1 style="margin: 0;">STAT-RESPIRA</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Deskripsi singkat
    st.markdown("""
    Selamat datang di aplikasi berbasis web untuk deteksi dini penyakit **Bronkopneumonia**
    melalui citra rontgen dada.
    """)

    st.markdown("""
    ### Metode yang Digunakan
    Aplikasi ini menggunakan model **Convolutional Neural Network (CNN)**
    dengan pendekatan **Transfer Learning** yang memanfaatkan arsitektur **MobileNetV2** 
    dalam melakukan klasifikasi citra rontgen.
    """)

    # Inventor rata kiri + bernomor
    st.subheader("Inventor:")
    st.markdown("""
    1. **Vikri Haikal**  
    2. **Muhammad Habib Mudafiq**  
    3. **Elsa Ika Rahmani**  
    """)

    # Universitas & Tahun rata tengah
    st.markdown(
        """
        <div style="text-align: center; margin-top: 20px;">
            <p><b>Universitas Diponegoro</b></p>
            <p><b>Tahun 2025</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )



# --- Bagian Halaman Deteksi ---
elif pilihan == "Deteksi":
    st.title("🔍 STAT-RESPIRA")
    st.write("Unggah gambar X-ray paru untuk diprediksi (Normal atau Bronkopneumonia).")

    # Load model
    model = "transfer_learning_mobilenetv2_model.keras"
    # Pastikan urutan class sesuai dengan saat training
    class_names = ["Normal", "Bronkopneumonia"]

    # Fungsi prediksi
    @st.cache_resource
    def predict_image(model, img, class_names, target_size=(224, 224)):
        img_resized = img.resize(target_size)
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class_idx = 1 if predictions[0][0] > 0.5 else 0
        predicted_class = class_names[predicted_class_idx]
        confidence = predictions[0][0] if predicted_class_idx == 1 else 1 - predictions[0][0]

        return predicted_class, confidence

    uploaded_file = st.file_uploader("Pilih file gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Gambar yang diunggah", use_column_width=True)

            if st.button("🩺 Diagnosa"):
                predicted_class, confidence = predict_image(model, img, class_names)
                st.markdown(f"### Hasil Prediksi: **{predicted_class}**")
                st.write(f"Tingkat keyakinan: **{confidence:.2%}**")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses gambar. Error: {e}")


