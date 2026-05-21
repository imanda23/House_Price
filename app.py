import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Judul Aplikasi
st.title("Aplikasi Prediksi Harga Rumah")
st.write("Aplikasi ini menggunakan model Robust Regression untuk mengestimasi harga rumah berdasarkan fitur yang dimasukkan.")

# 2. Fungsi untuk memuat resource
@st.cache_resource
def load_resources():
    with open('label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('robust_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return encoders, scaler, model

try:
    encoders, scaler, model = load_resources()
except Exception as e:
    st.error(f"Gagal memuat model atau scaler: {e}")
    st.stop()

# 3. Form Input User
st.sidebar.header("Input Fitur Rumah")

square_footage = st.sidebar.number_input("Luas Rumah (Square Footage)", min_value=100, max_value=10000, value=2500)
year_built = st.sidebar.number_input("Tahun Dibangun", min_value=1800, max_value=2025, value=2010)
lot_size = st.sidebar.number_input("Ukuran Tanah (Lot Size)", min_value=0.1, max_value=10.0, value=3.5)

num_bedrooms = st.sidebar.selectbox("Jumlah Kamar Tidur", options=encoders['Num_Bedrooms'].classes_)
num_bathrooms = st.sidebar.selectbox("Jumlah Kamar Mandi", options=encoders['Num_Bathrooms'].classes_)
garage_size = st.sidebar.selectbox("Kapasitas Garasi", options=encoders['Garage_Size'].classes_)
neighborhood_quality = st.sidebar.selectbox("Kualitas Lingkungan (1-10)", options=encoders['Neighborhood_Quality'].classes_)

# 4. Tombol Prediksi
if st.button("Prediksi Harga"):
    # Buat DataFrame dari input
    input_dict = {
        'Square_Footage': square_footage,
        'Num_Bedrooms': num_bedrooms,
        'Num_Bathrooms': num_bathrooms,
        'Year_Built': year_built,
        'Lot_Size': lot_size,
        'Garage_Size': garage_size,
        'Neighborhood_Quality': neighborhood_quality
    }
    df_input = pd.DataFrame([input_dict])

    # Preprocessing: Label Encoding
    for col in ['Num_Bedrooms', 'Num_Bathrooms', 'Garage_Size', 'Neighborhood_Quality']:
        df_input[col] = encoders[col].transform(df_input[col].astype(str))

    # Preprocessing: Scaling
    # Gunakan DataFrame agar nama fitur tetap terjaga untuk HuberRegressor
    data_scaled = pd.DataFrame(scaler.transform(df_input), columns=df_input.columns)

    # Prediksi
    prediction = model.predict(data_scaled)

    # Tampilkan Hasil
    st.success(f"### Estimasi Harga Rumah: ${prediction[0]:,.2f}")
    
st.info("Gunakan menu di samping untuk mengubah parameter rumah.")
