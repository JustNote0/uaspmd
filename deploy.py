import streamlit as st
import numpy as np
import pandas as pd

class SVM:
    def __init__(self):
        self.w = None
        self.b = None

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.where(linear_output >= 0, 1, 0)

params = np.load("svm_model_params.npz")
model = SVM()
model.w = params["w"]
model.b = params["b"]

st.set_page_config(page_title="Prediksi Penyakit Hati", layout="centered")

st.title("Prediksi Penyakit Hati Menggunakan SVM")

st.markdown("Masukkan data pasien berikut:")

age = st.number_input("Umur", min_value=1, max_value=100, step=1)
gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
tb = st.number_input("Total Bilirubin")
db = st.number_input("Direct Bilirubin")
alkphos = st.number_input("Alkaline Phosphotase")
sgpt = st.number_input("Alamine Aminotransferase (SGPT)")
sgot = st.number_input("Aspartate Aminotransferase (SGOT)")
tp = st.number_input("Total Proteins")
alb = st.number_input("Albumin")
ag_ratio = st.number_input("Albumin and Globulin Ratio")

if st.button("Prediksi"):
    gender_val = 1 if gender == "Laki-laki" else 0
    input_data = np.array([[age, gender_val, tb, db, alkphos, sgpt, sgot, tp, alb, ag_ratio]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Pasien kemungkinan mengalami penyakit hati.")
    else:
        st.success("Pasien kemungkinan tidak mengalami penyakit hati.")
